import argparse
import os

import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from transformers import T5Config

from data.sr_dataset import SeqRecDataset
from data.utils import ndcg_at_k, recall_at_k
from model.id.rqvae import RQVAE
from model.rec.t5 import T5Rec
from run_gr_id import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generative Recommendation Training")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--lr_rec", type=float, default=5e-4)
    parser.add_argument("--wd_rec", type=float, default=1e-2)
    parser.add_argument("--lr_id_load", type=float, default=1e-3)
    parser.add_argument("--wd_id_load", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=2e-2)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--id_mode", choices=["tiger", "letter"], default="tiger")
    parser.add_argument("--id_load_path", default="")
    parser.add_argument("--load_path", default="")
    parser.add_argument("--save_path", default="")
    parser.add_argument("--suffix_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--k_values", type=int, nargs="+", default=[10])
    parser.add_argument("--data_path", default="./data/")
    parser.add_argument("--dataset", default="Instruments")
    parser.add_argument("--max_his_len", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    if args.beam_size < max(args.k_values):
        parser.error("--beam_size must cover all requested K values.")
    return args


def create_dataloader(args):
    datasets = {
        mode: SeqRecDataset(args.data_path, args.dataset, args.max_his_len, mode)
        for mode in ["train", "valid", "test"]
    }
    return {
        mode: DataLoader(
            dataset,
            batch_size=args.batch_size if mode != "test" else max(1, args.batch_size // 4),
            shuffle=mode == "train",
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for mode, dataset in datasets.items()
    }


def initialize_models(args, device):
    if args.id_mode == "tiger":
        id_name = f"rqvae-{args.dataset}-tiger-lr_{args.lr_id_load}-wd_{args.wd_id_load}.pth"
    else:
        id_name = (
            f"rqvae-{args.dataset}-letter-alpha_{args.alpha}-beta_{args.beta}"
            f"-lr_{args.lr_id_load}-wd_{args.wd_id_load}.pth"
        )
    model_id = torch.load(
        args.id_load_path or os.path.join("ckpt", id_name),
        weights_only=False,
        map_location=device,
    )
    model_id.eval()
    model_id.requires_grad_(False)
    model_id.build_indices()
    sizes = model_id.codebook_size_list
    if len(sizes) != 4:
        raise ValueError("The agreed SID uses four quantization levels plus a suffix.")
    required_suffix_size = max(model_id.unique_indices) + 1
    if args.suffix_size < required_suffix_size:
        raise ValueError(
            f"Use the same --suffix_size >= {required_suffix_size} for both variants."
        )
    offsets = torch.tensor(
        [2 + sum(sizes[:level]) for level in range(len(sizes) + 1)],
        device=device,
    )
    model_id.token_indices = torch.tensor(
        model_id.full_all_indices, dtype=torch.long, device=device
    ) + offsets
    model_id.token_indices[0] = 0

    config = T5Config(
        vocab_size=sum(sizes) + args.suffix_size + 2,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        bos_token_id=0,
        d_model=128,
        d_ff=1024,
        d_kv=64,
        num_layers=4,
        num_decoder_layers=4,
        num_heads=6,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        feed_forward_proj="relu",
        use_cache=True,
    )
    return model_id, T5Rec(config).to(device)


def prepare_inputs(his, tgt, model_id):
    # Tokenization is a lookup in the frozen catalogue, independent of the batch.
    batch_size = his.size(0)
    input_ids = model_id.token_indices[his].flatten(1)
    eos_positions = input_ids.ne(0).sum(dim=1)
    input_ids = torch.cat([input_ids, input_ids.new_zeros(batch_size, 1)], dim=1)
    input_ids[torch.arange(batch_size, device=his.device), eos_positions] = 1
    labels = torch.cat(
        [model_id.token_indices[tgt], tgt.new_ones(batch_size, 1)], dim=1
    )
    return input_ids, labels, input_ids.ne(0).long()


def train_epoch(epoch, model_id, model_rec, dataloader, optimizer, device):
    model_rec.train()
    total_loss = 0.0
    for batch in dataloader:
        his, tgt = batch["his"].to(device), batch["tgt"].to(device)
        input_ids, labels, mask = prepare_inputs(his, tgt, model_id)
        optimizer.zero_grad()
        loss = model_rec(input_ids=input_ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(tgt)
    print(f"Epoch {epoch}: Training Loss = {total_loss / len(dataloader.dataset):.6f}")


@torch.no_grad()
def eval_epoch(model_id, model_rec, dataloader, device):
    model_rec.eval()
    total_loss = 0.0
    for batch in dataloader:
        his, tgt = batch["his"].to(device), batch["tgt"].to(device)
        input_ids, labels, mask = prepare_inputs(his, tgt, model_id)
        loss = model_rec(input_ids=input_ids, attention_mask=mask, labels=labels).loss
        total_loss += loss.item() * len(tgt)
    return {"loss": total_loss / len(dataloader.dataset)}


@torch.no_grad()
def test(model_id, model_rec, dataloader, device, beam_size=10, k_values=(10,)):
    model_rec.eval()
    catalogue = model_id.token_indices[1:].cpu().numpy()
    token_to_item = {tuple(code): item + 1 for item, code in enumerate(catalogue)}
    targets, predictions = [], []
    first_hits = 0
    for batch in dataloader:
        his, tgt = batch["his"].to(device), batch["tgt"].to(device)
        input_ids, labels, mask = prepare_inputs(his, tgt, model_id)
        pred_codes, first_tokens = model_rec._inference(
            input_ids=input_ids,
            attention_mask=mask,
            all_indices=catalogue,
            beam_size=beam_size,
        )
        pred_items = np.array([
            [token_to_item[tuple(code)] for code in sample]
            for sample in pred_codes.cpu().numpy()
        ])
        targets.append(tgt.cpu().numpy())
        predictions.append(pred_items)
        first_hits += (first_tokens == labels[:, 0]).sum().item()
    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(predictions, targets, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(predictions, targets, k)
    metrics["first_token_hit@1"] = first_hits / len(targets)
    metrics.update(model_id.sid_metrics)
    return metrics


def run():
    args = parse_args()
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda")
    dataloaders = create_dataloader(args)
    model_id, model_rec = initialize_models(args, device)
    optimizer_cls = AdamW if args.optimizer == "adamw" else Adam
    optimizer = optimizer_cls(
        model_rec.parameters(), lr=args.lr_rec, weight_decay=args.wd_rec
    )
    os.makedirs("ckpt", exist_ok=True)
    tag = f"-alpha_{args.alpha}-beta_{args.beta}" if args.id_mode == "letter" else ""
    ckpt_path = args.save_path or (
        f"ckpt/t5-{args.dataset}-{args.id_mode}{tag}"
        f"-lr_rec_{args.lr_rec}-wd_rec_{args.wd_rec}.pth"
    )
    print("Arguments:", vars(args))
    if not args.load_path:
        best_loss = float("inf")
        patience = 0
        for epoch in range(args.max_epochs):
            train_epoch(
                epoch, model_id, model_rec, dataloaders["train"], optimizer, device
            )
            valid = eval_epoch(model_id, model_rec, dataloaders["valid"], device)
            print(f"Epoch {epoch}: Validation Loss = {valid['loss']:.6f}")
            if valid["loss"] < best_loss:
                best_loss = valid["loss"]
                patience = 0
                torch.save({
                    "model_rec_weight": model_rec.state_dict(),
                    "item_tokens": model_id.token_indices.cpu(),
                    "sid_metrics": model_id.get_sid_metrics(),
                    "epoch": epoch,
                }, ckpt_path)
            else:
                patience += 1
            if patience >= args.patience:
                break
    checkpoint = torch.load(
        args.load_path or ckpt_path, weights_only=False, map_location=device
    )
    model_rec.load_state_dict(checkpoint["model_rec_weight"])
    model_id.token_indices = checkpoint["item_tokens"].to(device)
    model_id.sid_metrics = checkpoint["sid_metrics"]
    metrics = test(
        model_id, model_rec, dataloaders["test"], device, args.beam_size, args.k_values
    )
    print(
        f"Best Epoch {checkpoint['epoch']}: Test Metrics: "
        + ", ".join(f"{key} = {value:.6f}" for key, value in metrics.items())
    )


if __name__ == "__main__":
    run()

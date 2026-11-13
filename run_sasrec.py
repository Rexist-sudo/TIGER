import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.sr_dataset import SASRecDataset
from data.utils import ndcg_at_k, recall_at_k
from model.rec.sasrec import SASRec
from run_gr_id import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train SASRec and export item CF embeddings")
    parser.add_argument("--data_path", default="./data/")
    parser.add_argument("--dataset", default="Instruments")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--max_his_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--load_path", default="")
    parser.add_argument("--output_path", default="")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    predictions, targets = [], []
    for batch in dataloader:
        scores = model.predict(batch["his"].to(device))
        predictions.append((scores.topk(10, dim=-1).indices + 1).cpu().numpy())
        targets.append(batch["tgt"].numpy())
    predictions, targets = np.concatenate(predictions), np.concatenate(targets)
    return {"recall@10": recall_at_k(predictions, targets, 10),
            "ndcg@10": ndcg_at_k(predictions, targets, 10)}


def main():
    args = parse_args()
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda")
    datasets = {mode: SASRecDataset(args.data_path, args.dataset, args.max_his_len, mode)
                for mode in ["train", "valid", "test"]}
    loaders = {mode: DataLoader(dataset, batch_size=args.batch_size,
                               shuffle=mode == "train", num_workers=args.num_workers)
               for mode, dataset in datasets.items()}
    model = SASRec(datasets["train"].item_num, args.max_his_len,
                   dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    os.makedirs("ckpt", exist_ok=True)
    ckpt_path = f"ckpt/sasrec-{args.dataset}-32d.pth"
    print("Arguments:", vars(args))
    if not args.load_path:
        best_ndcg, patience = -1.0, 0
        for epoch in range(args.max_epochs):
            model.train()
            total_loss = 0.0
            for batch in loaders["train"]:
                optimizer.zero_grad()
                loss = model.loss(batch["his"].to(device), batch["pos"].to(device),
                                  batch["neg"].to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch["his"])
            valid = evaluate(model, loaders["valid"], device)
            print(f"Epoch {epoch}: loss={total_loss / len(datasets['train']):.6f}, "
                  + ", ".join(f"{key}={value:.6f}" for key, value in valid.items()), flush=True)
            if valid["ndcg@10"] > best_ndcg:
                best_ndcg, patience = valid["ndcg@10"], 0
                torch.save({"model_weight": model.state_dict(), "epoch": epoch}, ckpt_path)
            else:
                patience += 1
            if patience >= args.patience:
                break
    checkpoint = torch.load(args.load_path or ckpt_path, weights_only=False,
                            map_location=device)
    model.load_state_dict(checkpoint["model_weight"])
    metrics = evaluate(model, loaders["test"], device)
    print(f"Best Epoch {checkpoint['epoch']}: Test Metrics: "
          + ", ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
    output_path = args.output_path or os.path.join(
        args.data_path, args.dataset, args.dataset + ".cf-sasrec.npy"
    )
    np.save(output_path, model.item_embedding.weight[1:].detach().cpu().numpy())
    exposed = np.zeros(datasets["train"].item_num + 1, dtype=bool)
    for sample in datasets["train"].inter_data:
        exposed[sample["his"]] = True
        exposed[sample["pos"]] = True
    np.save(os.path.splitext(output_path)[0] + ".valid.npy", exposed[1:])
    print(f"Exported item embeddings to {output_path}")


if __name__ == "__main__":
    main()

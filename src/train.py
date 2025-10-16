import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os
import numpy as np
from tqdm import tqdm
from model import SimpleReLUMLP
from data import get_mnist_loaders
from utils import (
    flatten_params,
    power_iteration_hessian_max,
    layer_spectral_norms,
    compute_ntk_gram,
    topk_eigvals_numpy,
    hessian_topk_via_deflation,
)
from viz import make_live_animation


# === MSE loss version ===
def loss_fn_mse(model, inputs, targets):
    logits = model(inputs)
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, targets.view(-1, 1), 1.0)
    loss = nn.MSELoss()(logits, one_hot)
    return loss


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, y.view(-1, 1), 1.0)
            loss = criterion(logits, one_hot)
            batch = x.size(0)
            total_loss += loss.item() * batch
            total += batch
    return total_loss / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[512, 256, 128])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--ntk-subset", type=int, default=128, help="subset size for NTK computations"
    )
    parser.add_argument("--ntk-topk", type=int, default=5)
    parser.add_argument("--hessian-topk", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = SimpleReLUMLP(hidden_sizes=args.hidden_sizes).to(device)
    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size, train_size=args.train_size, test_size=args.test_size
    )
    opt = optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [],
        "test_loss": [],
        "sharpness": [],
        "layer_norms": [],
        "ntk_eigs": [],
        "hessian_eigs": [],
    }
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)

            # ✅ One-hot encode labels for MSE
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, y.view(-1, 1), 1.0)

            loss = nn.MSELoss()(logits, one_hot)
            loss.backward()
            opt.step()

            batch = x.size(0)
            running_loss += loss.item() * batch
            total += batch
        train_loss = running_loss / total
        test_loss = evaluate(model, test_loader, device)

        # === Sharpness estimate (top Hessian eigenvalue) ===
        xb, yb = next(iter(train_loader))
        xb = xb.to(device)
        yb = yb.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        try:
            lambda_max, _ = power_iteration_hessian_max(
                loss_fn_mse, model, params, xb, yb, n_iters=10, device=device
            )
        except Exception as e:
            print("Sharpness computation failed:", e)
            lambda_max = float("nan")

        # === Layer spectral norms ===
        try:
            norms = layer_spectral_norms(model)
        except Exception as e:
            print("Layer norm computation failed:", e)
            norms = {}

        # === NTK top-k ===
        try:
            ntk_subset = args.ntk_subset
            if ntk_subset > 0:
                xs = []
                ys = []
                for x, y in test_loader:
                    xs.append(x)
                    ys.append(y)
                    if sum([a.size(0) for a in xs]) >= ntk_subset:
                        break
                xs = torch.cat(xs, 0)[:ntk_subset].to(device)
                gram = compute_ntk_gram(model, xs, output_index=None, device=device)
                ntk_topk = topk_eigvals_numpy(gram, k=args.ntk_topk)
            else:
                ntk_topk = []
        except Exception as e:
            print("NTK computation failed:", e)
            ntk_topk = []

        # === Hessian top-k ===
        try:
            h_topk = hessian_topk_via_deflation(
                loss_fn_mse,
                model,
                params,
                xb,
                yb,
                k=args.hessian_topk,
                n_iters=10,
                device=device,
            )
        except Exception as e:
            print("Hessian top-k failed:", e)
            h_topk = []

        # === Record history ===
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["sharpness"].append(lambda_max)
        history["layer_norms"].append(norms)
        history["ntk_eigs"].append(
            list(map(float, ntk_topk)) if len(ntk_topk) > 0 else []
        )
        history["hessian_eigs"].append(
            list(map(float, h_topk)) if len(h_topk) > 0 else []
        )

        # Save intermediate history
        import json

        with open(os.path.join(results_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, test_loss={
                test_loss:.4f}, sharpness={lambda_max:.4f}"
        )

    # === Produce animation ===
    make_live_animation(history, results_dir=results_dir, fname="training_metrics.mp4")

    # Save final model
    torch.save(model.state_dict(), os.path.join(results_dir, "model_final.pt"))


if __name__ == "__main__":
    main()

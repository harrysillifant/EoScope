import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os
import numpy as np
from tqdm import tqdm
from model import SimpleReLUMLP
import json
from data import get_mnist_loaders, get_cifar10_loaders
from utils import (
    flatten_params,
    layer_spectral_norms,
    compute_ntk_gram,
    topk_eigvals_numpy,
    hessian_topk_via_deflation,
    num_linear_regions_basic,
    num_linear_regions_pier,
    top_hessian_eigenpair,
)
from viz import make_live_animation


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
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[200, 200])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--ntk-subset", type=int, default=128, help="subset size for NTK computations"
    )
    parser.add_argument("--ntk-topk", type=int, default=5)
    parser.add_argument("--hessian-topk", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default="gd")
    # "cifar10" is alternative
    parser.add_argument("--dataset", type=str, default="mnist")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.dataset == "mnist":
        train_loader, test_loader = get_mnist_loaders(
            batch_size=args.batch_size,
            train_size=args.train_size,
            test_size=args.test_size,
        )

        model = SimpleReLUMLP(hidden_sizes=args.hidden_sizes, input_dim=28 * 28).to(
            device
        )
    elif args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=args.batch_size,
            train_size=args.train_size,
            test_size=args.test_size,
        )
        model = SimpleReLUMLP(hidden_sizes=args.hidden_sizes, input_dim=3 * 32 * 32).to(
            device
        )
    else:
        raise Exception("Unknown dataset")

    if args.optimizer == "gd":
        opt = optim.SGD(model.parameters(), lr=args.lr)
        args.batch_size = None
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training on {args.dataset} dataset")

    history = {
        "train_loss": [],
        "test_loss": [],
        "sharpness": [],
        "layer_norms": [],
        "ntk_eigs": [],
        "hessian_eigs": [],
        "num_linear_regions_basic": [],
        "num_linear_regions_pier": [],
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
            preds = model(x)

            # if args.dataset == "mnist":
            one_hot = torch.zeros_like(preds)
            # turn mnist data into one hot for regression
            one_hot.scatter_(1, y.view(-1, 1), 1.0)
            y = one_hot

            loss = nn.MSELoss()(preds, y)
            loss.backward()
            opt.step()

            batch = x.size(0)
            running_loss += loss.item() * batch
            total += batch
        train_loss = running_loss / total
        test_loss = evaluate(model, test_loader, device)

        xb, yb = next(iter(train_loader))
        xb = xb.to(device)
        yb = yb.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        # Sharpness
        try:
            # lambda_max, _ = power_iteration_hessian_max(
            #     loss_fn_mse, model, params, xb, yb, n_iters=10, device=device
            # )
            lambda_max, v = top_hessian_eigenpair(
                model=model, criterion=loss_fn_mse, x_batch=xb, y_batch=yb, iters=20
            )
        except Exception as e:
            print("Sharpness computation failed:", e)
            lambda_max = float("nan")

        # Layer spectral norms
        try:
            raise ValueError
            norms = layer_spectral_norms(model)
        except Exception as e:
            print("Layer norm computation failed:", e)
            norms = {}

        # NTK top-k
        try:
            raise ValueError
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

        # Hessian top-k eigenvalues
        try:
            raise ValueError
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

        # Number Linear Regions basic activation pattern counting
        try:
            raise ValueError
            X = []
            for x, _ in train_loader:
                X.append(x)
            for x, _ in test_loader:
                X.append(x)
            X = torch.cat(X, dim=0)
            nlr_basic = num_linear_regions_basic(X=X, model=model, device=device)
        except Exception as e:
            print("Count linear regions failed:", e)
            nlr_basic = 0

        # Number of linear regions from Piers suggestion
        try:
            raise ValueError
            X, Y = [], []
            for x, y in train_loader:
                X.append(x)
                Y.append(y)
            for x, y in test_loader:
                X.append(x)
                Y.append(y)
            X = torch.cat(X, dim=0)
            Y = torch.cat(Y, dim=0)
            nlr_pier = num_linear_regions_pier(X=X, y=Y, model=model, device=device)
        except Exception as e:
            print("Pier count failed: ", e)
            nlr_pier = 0

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
        history["num_linear_regions_basic"].append(nlr_basic)
        history["num_linear_regions_pier"].append(nlr_pier)

        with open(os.path.join(results_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, test_loss={
                test_loss:.4f}, sharpness={lambda_max:.4f}, num_linear_regions_basic={
                nlr_basic
            }, num_linear_regions_pier={nlr_pier}"
        )

    make_live_animation(
        history, results_dir=results_dir, fname="training_metrics.mp4", lr=args.lr
    )

    torch.save(model.state_dict(), os.path.join(results_dir, "model_final.pt"))


if __name__ == "__main__":
    main()

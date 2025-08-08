import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# mix2smell imports
from mix2smell.data.data import TrainingData
from mix2smell.data.dataset import Mix2SmellData
from mix2smell.data.splits import SplitLoader
from mix2smell.data.collate import custom_collate

# TweedieLoss for point metrics
from pytorch_forecasting.metrics.point import TweedieLoss


def compute_mixture_embedding(x_features: torch.Tensor,
                              x_fractions: torch.Tensor,
                              num_solvents: int = 3) -> torch.Tensor:
    feat = x_features.squeeze(-1)
    weights = x_fractions[..., 0, :].squeeze(-1)
    mix_emb = (feat * weights.unsqueeze(-1)).sum(dim=1)
    mean_dil = weights.mean(dim=1, keepdim=True)
    solvents = x_fractions[..., 1, :].squeeze(-1).long()
    dominant, _ = solvents.mode(dim=1)
    sol_ohe = F.one_hot(dominant, num_classes=num_solvents).float()
    return torch.cat([mix_emb, mean_dil, sol_ohe], dim=1)


class MixtureNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, model_type):
        super().__init__()
        self.model_type = model_type
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(0.2)
        )
        if model_type == 'hurdle':
            self.classifier = torch.nn.Linear(hidden_dim, out_dim)
        self.regressor = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.Softplus())

    def forward(self, x):
        h = self.trunk(x)
        if self.model_type == 'hurdle':
            logits = self.classifier(h)
            probs = torch.sigmoid(logits)
            preds = self.regressor(h)
            return probs, preds
        return self.regressor(h)


def train_and_eval(args, p, split, device):
    # identical content as before...
    data = TrainingData()
    task = Mix2SmellData(dataset=data, task=['Task2_single', "Task2_mix"], featurization=args.features)
    splitter = SplitLoader(split_type='kfold')
    tr_idx, val_idx, _ = splitter(dataset_name=task.dataset.name, task=task.task,
                                   cache_dir=task.dataset.data_dir, split_num=split)
    tr_batch = next(iter(DataLoader(Subset(task, tr_idx.tolist()), batch_size=len(tr_idx), collate_fn=custom_collate)))
    va_batch = next(iter(DataLoader(Subset(task, val_idx.tolist()), batch_size=len(val_idx), collate_fn=custom_collate)))

    X_tr = compute_mixture_embedding(tr_batch['features'], tr_batch['fractions'], args.num_solvents).to(device)
    X_va = compute_mixture_embedding(va_batch['features'], va_batch['fractions'], args.num_solvents).to(device)
    y_tr = tr_batch['label'].to(device)
    y_va = va_batch['label'].to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    in_dim, out_dim = X_tr.shape[1], y_tr.shape[1]
    net = MixtureNet(in_dim, args.hidden, out_dim, args.model_type).to(device)

    if args.model_type == 'regression':
        reg_loss_fn = torch.nn.MSELoss()
    elif args.model_type == 'tweedie':
        reg_loss_fn = TweedieLoss(p=p, reduction='mean')
    else:
        class_loss_fn = torch.nn.BCEWithLogitsLoss()
        reg_loss_fn = torch.nn.MSELoss()

    optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val = float('inf')
    patience = 0

    for _ in range(args.epochs):
        net.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            if args.model_type == 'hurdle':
                probs, preds = net(Xb)
                mask = (yb > 0).float()
                loss = class_loss_fn(probs, mask) + reg_loss_fn(preds * mask, yb * mask)
            else:
                preds = net(Xb)
                loss = reg_loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            if args.model_type == 'hurdle':
                _, va_p = net(X_va)
                mask_va = (y_va > 0).float()
                val_loss = (class_loss_fn(_, mask_va) + reg_loss_fn(va_p * mask_va, y_va * mask_va)).item()
                y_pred = (va_p * mask_va)
            else:
                va_p = net(X_va)
                val_loss = reg_loss_fn(va_p, y_va).item()
                y_pred = va_p
            val_mae = mean_absolute_error(y_va.cpu().numpy(), y_pred.cpu().numpy())

        scheduler.step(val_loss)
        if val_mae < best_val:
            best_val = val_mae
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                break

    net.eval()
    with torch.no_grad():
        if args.model_type == 'hurdle':
            _, tr_p = net(X_tr)
            tr_pred = (tr_p * (y_tr > 0)).cpu().numpy()
            va_pred = y_pred.cpu().numpy()
        else:
            tr_pred = net(X_tr).cpu().numpy()
            va_pred = y_pred.cpu().numpy()

    # metrics
    train_mae = mean_absolute_error(y_tr.cpu().numpy(), tr_pred)
    val_mae = mean_absolute_error(y_va.cpu().numpy(), va_pred)
    train_pear = np.nanmean([pearsonr(y_tr.cpu().numpy()[:, i], tr_pred[:, i])[0] for i in range(out_dim)])
    val_pear = np.nanmean([pearsonr(y_va.cpu().numpy()[:, i], va_pred[:, i])[0] for i in range(out_dim)])
    train_cos = np.mean(cosine_similarity(y_tr.cpu().numpy(), tr_pred))
    val_cos = np.mean(cosine_similarity(y_va.cpu().numpy(), va_pred))

    return {
        'model_type': args.model_type,
        'p': p,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_pearson': train_pear,
        'val_pearson': val_pear,
        'train_cosine': train_cos,
        'val_cosine': val_cos,
        'y_true': y_va.cpu().numpy(),
        'y_pred': va_pred,
        'split': split
    }


def plot_predictions(results, num_cols=9, pad=1.2, out_dir='results/t2'):
    """Plot true vs. predicted values averaged across all splits in grid subplots, colored by split, and save to disk."""
    from collections import defaultdict
    results_dir = Path(out_dir)
    results_dir.mkdir(exist_ok=True)
    groups = defaultdict(lambda: defaultdict(lambda: []))
    for res in results:
        key = (res['model_type'], res['p'])
        groups[key]['true'].append((res['split'], res['y_true']))
        groups[key]['pred'].append((res['split'], res['y_pred']))

    cmap = plt.get_cmap('tab10')
    for (mt, p), data in groups.items():
        splits = sorted({s for s, _ in data['true']})
        num_labels = data['true'][0][1].shape[1]
        num_rows = int(np.ceil(num_labels / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.5, num_rows * 2.5))
        axes = axes.flatten()
        for i in range(num_labels):
            ax = axes[i]
            for idx, (split, true) in enumerate(data['true']):
                pred = data['pred'][idx][1]
                color = cmap(splits.index(split) % cmap.N)
                ax.scatter(true[:, i], pred[:, i], alpha=0.6, s=10, color=color,
                           label=f'Split {split}' if i == 0 else None)
            mn = min(min(t[:, i].min() for _, t in data['true']), min(p[:, i].min() for _, p in data['pred']))
            mx = max(max(t[:, i].max() for _, t in data['true']), max(p[:, i].max() for _, p in data['pred']))
            ax.plot([mn, mx], [mn, mx], 'k--', linewidth=0.5)
            ax.set_title(f"Label {i}", fontsize=6, pad=2)
            ax.set_xlabel('True', fontsize=5, labelpad=1)
            ax.set_ylabel('Pred', fontsize=5, labelpad=1)
            ax.tick_params(labelsize=4)
            if i == 0:
                ax.legend(fontsize=4, loc='upper left')
        for j in range(num_labels, len(axes)):
            axes[j].axis('off')
        title = f"{mt.upper()}" + (f"_p{p}" if mt == 'tweedie' else "")
        fig.suptitle(title.replace('.', '_'), fontsize=8)
        fig.tight_layout(pad=pad, rect=[0, 0.03, 1, 0.95])
        fname = f"{mt.upper()}" + (f"_p{p}" if mt == 'tweedie' else "")
        safe_fname = fname.replace('.', '_') + '.png'
        fig.savefig(results_dir / safe_fname, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='rdkit2d_normalized_features')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--num_solvents', type=int, default=3)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modes = ['regression', 'hurdle', 'tweedie']
    p_values = [1.2, 1.5, 1.8, 1.95]
    all_res = []

    for mt in modes:
        args.model_type = mt
        plist = p_values if mt == 'tweedie' else [0.0]
        for p in plist:
            for f in range(args.folds):
                all_res.append(train_and_eval(args, p, f, device))

    # print metrics
    df = pd.DataFrame(all_res)
    for mt in modes:
        print(f"\n== {mt.upper()} ==")
        sub = df[df['model_type'] == mt]
        metrics = ['train_mae', 'val_mae', 'train_pearson', 'val_pearson', 'train_cosine', 'val_cosine']
        if mt == 'tweedie':
            for p in p_values:
                block = sub[sub['p'] == p][metrics]
                stats = block.agg(['mean', 'std'])
                print(f"-- p={p} --")
                for m in metrics:
                    mu, sd = stats.loc['mean', m], stats.loc['std', m]
                    print(f"{m}: {mu:.4f} ± {sd:.4f}")
        else:
            stats = sub[metrics].agg(['mean', 'std'])
            for m in metrics:
                mu, sd = stats.loc['mean', m], stats.loc['std', m]
                print(f"{m}: {mu:.4f} ± {sd:.4f}")

    # plot aggregated predictions per split in color and save
    plot_predictions(all_res, pad=1.5)

if __name__ == '__main__':
    main()

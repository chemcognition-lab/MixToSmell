import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from xgboost import XGBRegressor
import wandb
import ast
import itertools

from mix2smell.data.dataset import Mix2SmellData
from mix2smell.data.collate import custom_collate
from mix2smell.model.utils import compute_key_padding_mask
from mix2smell.model.aggregation import PrincipalNeighborhoodAggregation
from mix2smell.data.utils import UNK_TOKEN
from mix2smell.data.splits import SplitLoader
from mixture_data_loading import MixBySingle, MixByAllSingle

# === Load selected features mask ===
mask_path = "selected_features_with_mixrata.npy"
selected_mask = np.load(mask_path)
print("✅ Loaded selected feature mask:")
print(f"  → Total features: {selected_mask.shape[0]}, Selected: {np.sum(selected_mask)}")


# === Utilities ===
def extract_mix_rata_from_dataset(dataset, indices):
    mix_rata_list = []
    for idx in indices:
        sample_data = dataset.inputs.iloc[idx]
        mix_rata_str = sample_data["mix_rata"]
        mix_rata = ast.literal_eval(mix_rata_str) if isinstance(mix_rata_str, str) else mix_rata_str
        mix_rata = mix_rata[:53] if len(mix_rata) > 53 else mix_rata + [0.0] * (53 - len(mix_rata))
        mix_rata_list.append(mix_rata)
    return torch.tensor(mix_rata_list, dtype=torch.float32)


def compute_mixture_embedding_with_rata(x_features, x_fractions, mix_rata):
    mix_embs = []
    for i, feat in enumerate(torch.unbind(x_features, dim=-1)):
        mask = compute_key_padding_mask(feat, UNK_TOKEN)
        frac = x_fractions[:, :, :, i]
        mol_emb = torch.concat([feat, frac], dim=-1)
        mix_rata_exp = mix_rata.unsqueeze(1).expand(feat.size(0), feat.size(1), -1)
        mol_emb = torch.concat([mol_emb, mix_rata_exp], dim=-1)
        emb = PrincipalNeighborhoodAggregation()(mol_emb, mask)
        mix_embs.append(emb)

    stacked = torch.stack(mix_embs, dim=-1)
    if stacked.shape[2] > 1:
        b, f, m, _ = stacked.shape
        stacked = stacked.reshape(b, f * m, 1)
    return stacked


def evaluate_model(X, y, model):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred, multioutput="uniform_average")
    cs = np.mean(cosine_similarity(y, y_pred))

    rowwise_pearsons = [pearsonr(y[i, :], y_pred[i, :])[0] for i in range(len(y))
                        if not np.all(y[i, :] == 0) and not np.all(y_pred[i, :] == 0)]
    rowwise_pearson = np.nanmean(rowwise_pearsons)

    # Reduced Pearson: average Pearson correlation across all outputs
    reduced_pearsons = [pearsonr(y[:, j], y_pred[:, j])[0] for j in range(y.shape[1])]
    reduced_pearson = np.nanmean(reduced_pearsons)

    return mae, r2, cs, rowwise_pearson, reduced_pearson


def extract_train_val_data():
    dataset = MixBySingle()
    task = Mix2SmellData(
        dataset=dataset,
        task=["Task2_mix"],
        featurization="rdkit2d_normalized_features",
    )
    train_idx, val_idx, _ = SplitLoader(
        dataset_name=task.dataset.name,
        task=task.task,
        cache_dir=task.dataset.data_dir,
    )()
    train_ds = Subset(task, train_idx.tolist())
    val_ds   = Subset(task, val_idx.tolist())
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), collate_fn=custom_collate)
    val_loader   = DataLoader(val_ds,   batch_size=len(val_ds),   collate_fn=custom_collate)

    train_batch = next(iter(train_loader))
    val_batch   = next(iter(val_loader))

    train_mix_rata = extract_mix_rata_from_dataset(task.dataset, train_idx.tolist())
    val_mix_rata   = extract_mix_rata_from_dataset(task.dataset, val_idx.tolist())

    train_emb = compute_mixture_embedding_with_rata(train_batch["features"], train_batch["fractions"], train_mix_rata)
    val_emb   = compute_mixture_embedding_with_rata(val_batch["features"],   val_batch["fractions"],   val_mix_rata)

    train_X = train_emb.squeeze(-1).numpy()[:, selected_mask]
    train_y = train_batch["label"].numpy()
    val_X   = val_emb.squeeze(-1).numpy()[:, selected_mask]
    val_y   = val_batch["label"].numpy()
    return train_X, train_y, val_X, val_y


if __name__ == "__main__":
    # Define ranges for grid search
    param_grid = {
        "n_estimators": [500],
        "max_depth": [3],
        "learning_rate": [0.01],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [1],
        "gamma": [0],
        "reg_lambda": [1],
        "reg_alpha": [0],
        "eval_metric": ["mae", "rmse", "logloss"],
        "objective": ["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"]
    }

    all_param_names = list(param_grid.keys())
    all_param_combinations = list(itertools.product(*param_grid.values()))

    train_X, train_y, val_X, val_y = extract_train_val_data()
    curr_max_pearson = 0

    for param_values in all_param_combinations:
        config = dict(zip(all_param_names, param_values))

        wandb.init(project="mix2smell-xgb-eval", config=config)

        model = XGBRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            min_child_weight=config["min_child_weight"],
            gamma=config["gamma"],
            reg_lambda=config["reg_lambda"],
            reg_alpha=config["reg_alpha"],
            verbosity=0,
            eval_metric=config["eval_metric"],
            objective=config["objective"],
        )
        model.fit(train_X, train_y)

        mae_val, r2_val, cs_val, row_pearson_val, reduced_pearson_val = evaluate_model(val_X, val_y, model)

        print("=" * 60)
        print(f"[ValSet] Config: {config}")
        print(f"→ MAE: {mae_val:.4f}, R2: {r2_val:.4f}, CosSim: {cs_val:.4f}, RowPearson: {row_pearson_val:.4f}, ReducedPearson: {reduced_pearson_val:.4f}")

        wandb.log({
            "val/mae": mae_val,
            "val/r2": r2_val,
            "val/cosine_similarity": cs_val,
            "val/rowwise_pearson": row_pearson_val,
            "val/reduced_pearson": reduced_pearson_val,
        })
        if row_pearson_val > curr_max_pearson:
            model.save_model("final_model.json")
            curr_max_pearson = row_pearson_val
        wandb.finish()

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

import pandas as pd

from mix2smell.data.dataset import Mix2SmellData
from mix2smell.data.data import TestData
from mix2smell.data.collate import custom_collate
from mix2smell.model.utils import compute_key_padding_mask
from mix2smell.model.aggregation import PrincipalNeighborhoodAggregation
from mix2smell.data.utils import UNK_TOKEN
from mix2smell.data.splits import SplitLoader
from mixture_data_loading import MixBySingleTest, MixByAllSingle


# === Load selected features mask ===
mask_path = "selected_features_with_mixrata.npy"
selected_mask = np.load(mask_path)
print("✅ Loaded selected feature mask:")
print(f"  → Total features: {selected_mask.shape[0]}, Selected: {np.sum(selected_mask)}")


# === Utilities ===
def extract_mix_rata_from_dataset(dataset):
    mix_rata_list = []
    for idx in range(130):
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


model = XGBRegressor()
model.load_model("final_model.json")

dataset = MixBySingleTest()

task = Mix2SmellData(
    dataset=dataset,
    task=["Task2_mix"],
    featurization="rdkit2d_normalized_features",
)

loader = DataLoader(task, batch_size=len(task), collate_fn=custom_collate)

batch = next(iter(loader))

test_mix_rata = extract_mix_rata_from_dataset(task.dataset)
                                              
emb = compute_mixture_embedding_with_rata(batch["features"], batch["fractions"], test_mix_rata)

X = emb.squeeze(-1).numpy()[:, selected_mask]
y = batch["label"].numpy()

pred = pd.DataFrame(model.predict(X))

pred.to_csv("../results/t2_predictions.csv")
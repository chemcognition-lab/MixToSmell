import os

os.environ["WANDB_SILENT"] = "true"

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import ast
import wandb

from scipy.stats import pearsonr
from mix2smell.data.data import TrainingData
from mix2smell.data.dataset import Mix2SmellData
from mix2smell.data.splits import SplitLoader
from mix2smell.data.utils import UNK_TOKEN
from mix2smell.model.utils import compute_key_padding_mask
from mix2smell.model.aggregation import PrincipalNeighborhoodAggregation
from mix2smell.data.collate import custom_collate
from t2_scripts.mixture_data_loading import MixBySingle, MixByAllSingle
import ast

from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import RFECV


def compute_mixture_embedding_with_rata(x_features, x_fractions, mix_rata):
    """
    Compute mixture embeddings including mix_rata as additional features
    
    Args:
        x_features: [batch, molecules, rdkit_features, 1] - RDKit molecular features
        x_fractions: [batch, molecules, fraction_features, 1] - Dilution/fraction info
        mix_rata: [batch, 53] - Predicted mixture RATA from single molecules
    
    Returns:
        stacked: [batch, final_features, 1] - Combined embeddings
    """
    print(f"Input shapes - features: {x_features.shape}, fractions: {x_fractions.shape}, mix_rata: {mix_rata.shape}")
    
    mix_embs = []
    for i, feat in enumerate(torch.unbind(x_features, dim=-1)):
        mask = compute_key_padding_mask(feat, UNK_TOKEN)
        frac = x_fractions[:, :, :, i]
        mol_emb = torch.concat([feat, frac], dim=-1)
        
        # Add mix_rata as additional features for each molecule
        batch_size, num_molecules = feat.shape[:2]
        mix_rata_expanded = mix_rata.unsqueeze(1).expand(batch_size, num_molecules, -1)

        # Final concatenation: [RDKit + fractions + predicted_rata]
        mol_emb = torch.concat([mol_emb, mix_rata_expanded], dim=-1)
       
        emb = PrincipalNeighborhoodAggregation()(mol_emb, mask)
        mix_embs.append(emb)
    
    stacked = torch.stack(mix_embs, dim=-1)
    if stacked.shape[2] > 1:
        b, f, m, _ = stacked.shape
        stacked = stacked.reshape(b, f * m, 1)
    
    print(f"Final embedding shape: {stacked.shape}")
    return stacked

def extract_mix_rata_from_dataset(dataset, indices):
    """Extract mix_rata for given indices from the dataset"""
    mix_rata_list = []
    
    for idx in indices:
        sample_data = dataset.inputs.iloc[idx]
        mix_rata_str = sample_data["mix_rata"]
        
        # Parse the string representation of the list
        if isinstance(mix_rata_str, str):
            mix_rata = ast.literal_eval(mix_rata_str)
        else:
            mix_rata = mix_rata_str
            
        # Ensure it's exactly 53 elements
        if len(mix_rata) != 53:
            print(f"Warning: mix_rata has {len(mix_rata)} elements, expected 53. Adjusting...")
            if len(mix_rata) > 53:
                mix_rata = mix_rata[:53]  # Truncate
            else:
                mix_rata = mix_rata + [0.0] * (53 - len(mix_rata))  # Pad with zeros
        
        mix_rata_list.append(mix_rata)
    
    return torch.tensor(mix_rata_list, dtype=torch.float32)

# ————— initialize W&B run —————
wandb.init(
    project="mix2smell-nested_xgboost",
    name="hybrid-xgb-t3", ## RENAME 
    config={
        "threshold_method": "percentile",
        "threshold_percentile": 50,
        "clf_n_estimators": 250,
        "clf_max_depth": 250,
        "clf_lr": 0.01,
        "reg_n_estimators": 250,
        "reg_max_depth": 250,
        "reg_lr": 0.01,
        "use_mix_rata": True,  # Track that we're using mix_rata
    },
    settings=wandb.Settings(console="off")
)

# ————— load data & compute embeddings —————
print("Loading MixBySingle dataset...")
data = MixBySingle()

task = Mix2SmellData(
    dataset=data,
    task=["Task2_mix"], # use only Task2 mix features
    featurization="rdkit2d_normalized_features",
)

print(f"Dataset size: {len(task)}")

# Load splits
train_idx, val_idx, test_idx = SplitLoader(# k‐fold split
    dataset_name=task.dataset.name,
    task=task.task,
    cache_dir=task.dataset.data_dir)()

train_idx = torch.concat([train_idx, val_idx])
val_idx   = test_idx

train_ds = Subset(task, train_idx.tolist())
val_ds   = Subset(task, val_idx.tolist())

train_loader = DataLoader(train_ds, batch_size=len(train_ds), collate_fn=custom_collate)
val_loader   = DataLoader(val_ds,   batch_size=len(val_ds),   collate_fn=custom_collate)

batch = next(iter(train_loader))

# Extract mix_rata for training samples
print("Extracting mix_rata for training samples...")
train_mix_rata = extract_mix_rata_from_dataset(task.dataset, train_idx.tolist())

# Compute embeddings WITH mix_rata
print("Computing training embeddings with mix_rata...")
train_X = compute_mixture_embedding_with_rata(
    batch["features"], 
    batch["fractions"],
    train_mix_rata
).squeeze(-1).numpy()
train_y = batch["label"].numpy()

train_embs = torch.load("dataset/processed/set_embeddings/emb_all_split4_train.pt")
val_embs = torch.load("dataset/processed/set_embeddings/emb_all_split4_val.pt")

batch = next(iter(val_loader))

val_mix_rata = extract_mix_rata_from_dataset(task.dataset, val_idx.tolist())

# Compute embeddings with mix_rata
val_X = compute_mixture_embedding_with_rata(
    batch["features"], 
    batch["fractions"],
    val_mix_rata
).squeeze(-1).numpy()
val_y = batch["label"].numpy()

n_samples, n_outputs = train_y.shape

train_X = torch.cat((torch.tensor(train_X), train_embs), dim = 1).numpy()
val_X = torch.cat((torch.tensor(val_X), val_embs), dim = 1).numpy()


print(f"Train X={train_X.shape}, y={train_y.shape}")
print(f" Val  X={  val_X.shape}, y={  val_y.shape}")

# ————— define thresholds & binary masks —————
method = wandb.config.get("threshold_method", "median")
if method == "median":
    thresholds = np.median(train_y, axis=0)
elif method == "percentile":
    pct = wandb.config.get("threshold_percentile", 50)
    thresholds = np.percentile(train_y, pct, axis=0)
else:
    raise ValueError(f"Unknown threshold_method: {method}")

train_mask = train_y > thresholds[np.newaxis, :]
val_mask   = val_y   > thresholds[np.newaxis, :]

# ————— Feature Selection —————
print("\n=== Feature Selection ===")
selected_features_per_output = {}
train_X_selected = train_X.copy()
val_X_selected = val_X.copy()

print("Performing feature selection...")
for j in range(n_outputs):
    if train_mask[:, j].sum() < 10:  # Skip if too few positive samples
        continue
        
    # Use a simple XGB for feature selection
    selector = RFECV(
        estimator=XGBRegressor(n_estimators=50, max_depth=3, random_state=42),
        step=10,
        cv=3,
        scoring='neg_mean_absolute_error',
        min_features_to_select=20
    )
    
    # Fit on positive samples for this output
    positive_idx = np.where(train_mask[:, j])[0]
    if len(positive_idx) >= 10:
        print(f"  Processing output {j} with {len(positive_idx)} positive samples...")
        selector.fit(train_X[positive_idx], train_y[positive_idx, j])
        selected_features_per_output[j] = selector.support_
        
        print(f"  Output {j}: Selected {selector.support_.sum()}/{len(selector.support_)} features")

# Combine feature selections (union approach)
if selected_features_per_output:
    combined_mask = np.any(list(selected_features_per_output.values()), axis=0)
    train_X_selected = train_X[:, combined_mask]
    val_X_selected = val_X[:, combined_mask]
    print(f"Combined: Using {combined_mask.sum()}/{len(combined_mask)} features")
    
    # Log feature selection info
    wandb.log({
        "feature_selection/original_features": train_X.shape[1],
        "feature_selection/selected_features": combined_mask.sum(),
        "feature_selection/reduction_ratio": combined_mask.sum() / train_X.shape[1]
    })
else:
    print("No feature selection performed")

# Update X variables for the rest of the pipeline
train_X = train_X_selected
val_X = val_X_selected

print(f"Final training shape: {train_X.shape}")

# ————— Stage 1: train one classifier per output —————
classifiers     = []
train_mask_pred = np.zeros_like(train_mask, dtype=bool)
val_mask_pred   = np.zeros_like(val_mask,   dtype=bool)

for j in range(n_outputs):
    yj = train_mask[:, j]
    pos = int(yj.sum())
    neg = n_samples - pos
    if pos < 5:
        classifiers.append(None)
        continue

    print(f"Training classifier {j}: {pos} positive, {neg} negative samples")
    
    clf = XGBClassifier(
        n_estimators=wandb.config.clf_n_estimators,
        max_depth=wandb.config.clf_max_depth,
        learning_rate=wandb.config.clf_lr,
        use_label_encoder=False,
        scale_pos_weight=neg / pos,
        verbosity=0,
        eval_metric="logloss",
    )
    clf.fit(
        train_X, yj,
        eval_set=[(train_X, yj), (val_X, val_mask[:, j])],
        verbose=False,
    )
    classifiers.append(clf)
    train_mask_pred[:, j] = clf.predict(train_X)
    val_mask_pred[:,   j] = clf.predict(val_X)

    # log classifier metrics
    res = clf.evals_result()
    wandb.log({
        f"classifier/{j}_train_logloss": res['validation_0']['logloss'][-1],
        f"classifier/{j}_val_logloss":   res['validation_1']['logloss'][-1],
    })

# ————— Stage 2: train regressors on masked‐on samples —————
regressors = []
for j in range(n_outputs):
    idx = np.where(train_mask[:, j])[0]
    if len(idx) < 5:
        regressors.append(None)
        continue

    print(f"Training regressor {j}: {len(idx)} positive samples")
    
    reg = XGBRegressor(
        n_estimators=wandb.config.reg_n_estimators,
        max_depth=wandb.config.reg_max_depth,
        learning_rate=wandb.config.reg_lr,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
        eval_metric="mae",
    )
    reg.fit(
        train_X[idx], train_y[idx, j],
        eval_set=[(train_X[idx], train_y[idx, j]), (val_X, val_y[:, j])],
        verbose=False,
    )
    regressors.append(reg)

    # log regressor metrics
    res = reg.evals_result()
    wandb.log({
        f"regressor/{j}_train_mae": res['validation_0']['mae'][-1],
        f"regressor/{j}_val_mae":   res['validation_1']['mae'][-1],
    })

# ————— assemble hybrid predictions —————
def hybrid_predict(X, mask_pred, regressors):
    n, d = mask_pred.shape
    Y = np.zeros((n, d))
    for j, reg in enumerate(regressors):
        if reg is None:
            continue
        yj = reg.predict(X)
        Y[:, j] = np.where(mask_pred[:, j], yj, 0.0)
    return Y

train_hat = hybrid_predict(train_X, train_mask_pred, regressors)
val_hat   = hybrid_predict(val_X,   val_mask_pred,   regressors)

# ————— compute & log final metrics —————
tr_acc = accuracy_score(train_mask.flatten(), train_mask_pred.flatten())
vl_acc = accuracy_score(val_mask.flatten(),   val_mask_pred.flatten())

train_mae = mean_absolute_error(train_y, train_hat)
val_mae   = mean_absolute_error(val_y,   val_hat)
train_r2  = r2_score(train_y, train_hat, multioutput="uniform_average")
val_r2    = r2_score(val_y,   val_hat,   multioutput="uniform_average")

cs_train = np.mean(cosine_similarity(train_y, train_hat))
cs_val   = np.mean(cosine_similarity(val_y,   val_hat))

# ————— Pearson correlation —————
pearson_train = [pearsonr(train_y[:, j], train_hat[:, j])[0] for j in range(n_outputs)]
pearson_val   = [pearsonr(val_y[:,   j],   val_hat[:,   j])[0] for j in range(n_outputs)]
avg_pearson_train = np.nanmean(pearson_train)
avg_pearson_val   = np.nanmean(pearson_val)

print("\n== Final Hybrid Metrics ==")
print(f"Mask Accuracy:   train={tr_acc:.3f}, val={vl_acc:.3f}")
print(f"Hybrid MAE:      train={train_mae:.4f}, val={val_mae:.4f}")
print(f"Hybrid R²:       train={train_r2:.4f}, val={val_r2:.4f}")
print(f"Avg CosineSim:   train={cs_train:.4f}, val={cs_val:.4f}")
print(f"Avg Pearson  r:  train={avg_pearson_train:.4f}, val={avg_pearson_val:.4f}")

wandb.log({
    "mask/train_acc":  tr_acc,
    "mask/val_acc":    vl_acc,
    "reg/train_mae":   train_mae,
    "reg/val_mae":     val_mae,
    "reg/train_r2":    train_r2,
    "reg/val_r2":      val_r2,
    "cs/train":        cs_train,
    "cs/val":          cs_val,
    "pearson/train":   avg_pearson_train,
    "pearson/val":     avg_pearson_val,
})

# Save feature selection results
if selected_features_per_output:
    np.save("t2_scripts/selected_features_with_mixrata_all.npy", combined_mask)
    print(f"Saved feature selection mask to t2_scripts/selected_features_with_mixrata_all.npy")

wandb.finish()
print("Training completed!")
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader

# Append src path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mix2smell.data.data import TrainingData
from mix2smell.data.dataset import Mix2SmellData
from mix2smell.data.collate import custom_collate

# Paths
CONFIG_PATH = "t2_scripts/config/xgb_model_config.yaml"
EMBEDDING_PATH = "t2_scripts/Embeddings/embeddings.npz"
TASK = "Task2_mix"
BATCH_SIZE = 128

# Load labels
raw = TrainingData()
ds = Mix2SmellData(dataset=raw, task=[TASK], featurization="rdkit2d_normalized_features")
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

y_list = []
for batch in loader:
    labels = batch["label"]  # shape [B, 51]
    y_list.append(labels.numpy())
y = np.concatenate(y_list, axis=0)  # final shape [N, 51]

# Load embeddings
print("Loading POM embeddings from .npz...")
embedding_data = np.load(EMBEDDING_PATH)
X = embedding_data["embeddings"]

assert X.shape[0] == y.shape[0], f"Embedding and label sample count mismatch: {X.shape[0]} != {y.shape[0]}"

# Load model config
def get_model(single_output=False):
    # Load config if available
    if os.path.exists(CONFIG_PATH):
        config = OmegaConf.load(CONFIG_PATH)
        model_type = config.get("model_type", "xgboost").lower()
        model_params = config.get("model_params", {})

        if model_type == "xgboost":
            model = XGBRegressor(**model_params)
        elif model_type == "catboost":
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(**model_params, verbose=0)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    else:
        print(f"Config file {CONFIG_PATH} not found. Using default xgboost.")
        model = XGBRegressor(n_estimators=100, random_state=42)

    return model if single_output else MultiOutputRegressor(model)


# RATA labels
rata_labels = ["Green", "Cucumber", "Herbal", "Mint", "Woody", 
               "Pine", "Floral", "Powdery", "Fruity", "Citrus", "Tropical", "Berry", "Peach", "Sweet", 
               "Caramellic", "Vanilla", "BrownSpice", "Smoky", "Burnt", "Roasted", "Grainy", "Meaty", "Nutty", 
               "Fatty", "Coconut", "Waxy", "Dairy", "Buttery", "Cheesy", "Sour", "Fermented", "Sulfurous", "Garlic.Onion", 
               "Earthy", "Mushroom", "Musty", "Ammonia", "Fishy", "Fecal", "Rotten.Decay", 
               "Rubber", "Phenolic", "Animal", "Medicinal", "Cooling", "Sharp", "Chlorine", "Alcoholic", "Plastic", 
               "Ozone", "Metallic"]

# Baseline per-label feature selection using RFECV
from collections import defaultdict

print("\n=== Running RFECV for All RATA Labels ===")
cv_options = [3, 5, 10]
step_options = [10]  # You can expand this if needed

# Store results
label_supports = {}
label_metrics = defaultdict(list)

for i, label_name in enumerate(rata_labels):
    print(f"\n--- Label {i}: {label_name} ---")
    y_single = y[:, i]
    X_train, X_test, y_train, y_test = train_test_split(X, y_single, test_size=0.2, random_state=42)

    best_pearson = -1.0
    best_mask = None

    for cv in cv_options:
        for step in step_options:
            print(f"Trying RFECV(cv={cv}, step={step})")
            model = get_model(single_output=True)

            rfecv = RFECV(
                estimator=model,
                step=step,
                cv=cv,
                scoring='neg_mean_absolute_error',
                verbose=0
            )

            try:
                rfecv.fit(X_train, y_train)
                y_pred = rfecv.predict(X_test)

                pearson_corr = pearsonr(y_test, y_pred)[0]
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                if pearson_corr > best_pearson:
                    best_pearson = pearson_corr
                    best_mask = rfecv.support_

                print(f"Pearson: {pearson_corr:.4f} | R²: {r2:.4f} | MAE: {mae:.4f}")

            except Exception as e:
                print(f"Failed on {label_name} with cv={cv}, step={step}: {e}")

    label_supports[label_name] = best_mask
    label_metrics["label"].append(label_name)
    label_metrics["n_features"].append(np.sum(best_mask))
    label_metrics["pearson"].append(best_pearson)

# Save individual supports
support_save_path = "t2_scripts/selected_features_per_label.npy"
np.save(support_save_path, label_supports)
print(f"Saved per-label feature supports to: {support_save_path}")

summary_df = pd.DataFrame(label_metrics)
summary_df.to_csv("t2_scripts/label_feature_selection_summary.csv", index=False)
print("Saved summary metrics per label.")

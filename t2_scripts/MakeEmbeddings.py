import torch
import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mix2smell.model.model_builder import build_pom_model
from mix2smell.data.dataset import Mix2SmellData
from mix2smell.data.data import TrainingData, KellerData, TrainingDataIntensity, TrainingDataAll, GSLFData
from mix2smell.data.collate import custom_collate
from mix2smell.data.utils import UNK_TOKEN


class POMEmbeddingExtractor:
    def __init__(self, config_path: str, dataset_type: str = "training", batch_size: int = 64, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load YAML config
        self.config = OmegaConf.load(config_path)

        # Temporarily patch torch.load inside build_pom_model to include map_location
        torch_load_original = torch.load
        torch.load = lambda *args, **kwargs: torch_load_original(*args, map_location=self.device, **kwargs)

        self.model = build_pom_model(self.config.pom_model).to(self.device)
        self.model.eval()

        # Restore torch.load to its original form
        torch.load = torch_load_original

        # Load dataset and DataLoader
        self.dataset = self._load_dataset(dataset_type)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            collate_fn=custom_collate,
            shuffle=False
        )

    def _load_dataset(self, dataset_type: str, task: str = ["Task2_mix"]):
        dataset_type = dataset_type.lower()
        if dataset_type == "keller":
            raw_data = KellerData()
        elif dataset_type == "training":
            raw_data = TrainingData()
        elif dataset_type == "trainingintensity":
            raw_data = TrainingDataIntensity()
        elif dataset_type == "trainingall":
            raw_data = TrainingDataAll()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        featurization = self.config.dataset.featurization
        return Mix2SmellData(dataset=raw_data, task=task, featurization=featurization)

    def generate_embeddings(self):
        all_embeddings = []
        with torch.no_grad():
            for batch in self.dataloader:
                x = batch['features'].to(self.device)
                ids = batch['ids'].to(self.device)
                fractions = batch['fractions'].to(self.device)
                emb = self.model.embed_with_fractions(x, ids, fractions)
                # Create mask and compute average
                ids = ids.squeeze(-1)  # shape [B, M]
                padding_mask = (ids != UNK_TOKEN).unsqueeze(-1)  # shape [B, M, 1]

                emb_masked = emb * padding_mask  # shape (B, M, D)
                emb_sum = emb_masked.sum(dim=1)  # shape (B, D)
                emb_count = padding_mask.sum(dim=1)  # shape (B, 1)
                mixture_emb = emb_sum / emb_count.clamp(min=1)  # avoid division by zero

                all_embeddings.append(mixture_emb.cpu())
        final_emb = torch.cat(all_embeddings).numpy()
        print(f"[INFO] Final embeddings shape: {final_emb.shape}")
        return final_emb


def main():
    parser = argparse.ArgumentParser(description="Generate single-molecule embeddings using POM model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--dataset", type=str, default="keller", choices=["keller", "training", "trainingintensity", "trainingall"], help="Dataset type to load.")
    parser.add_argument("--task", type=str, required=True, choices=["keller", "Task1", "Task2_single", "Task2_mix"], help="Task name for loading.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output .npz file.")

    args = parser.parse_args()

    extractor = POMEmbeddingExtractor(
        config_path=args.config,
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        device=args.device
    )
    embeddings = extractor.generate_embeddings()
    output_path = Path("t2_scripts/Embeddings") / Path(args.output).with_suffix(".npz").name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, embeddings=embeddings)
    print(f"[INFO] Embeddings saved to {output_path}")


if __name__ == "__main__":
    main()

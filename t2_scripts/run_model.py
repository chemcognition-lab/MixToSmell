import argparse
import os
import torch
import sys
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchtune.training.metric_logging import WandBLogger
from omegaconf import OmegaConf

from mix2smell.data.dataset import Mix2SmellData
from mix2smell.data.data import DATA_CATALOG
from mix2smell.data.featurization import FEATURIZATION_TYPE
from mix2smell.data.collate import custom_collate
from mix2smell.data.splits import SplitLoader

from mix2smell.model.train import LOSS_MAP, log_metrics, one_epoch
from mix2smell.model.model_builder import build_mixture_model

from typing import Optional
import torchmetrics
import tqdm
from mix2smell.model.utils import EarlyStopping
from mix2smell.model.custom_metrics import SafeMultilabelAUROC, ReducedPearsonCorrCoef, ReducedKendallRankCorrCoef


def train(
    root_dir: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_type: str,
    lr_mol_encoder: float,
    lr_other: float,
    device,
    weight_decay: float,
    max_epochs: int,
    patience: Optional[int] = None,
    experiment_name: Optional[str] = None,
    wandb_logger: Optional[WandBLogger] = None,
):
    loss_fn = LOSS_MAP[loss_type]()

    metrics = torchmetrics.MetricCollection(
        [
            ReducedPearsonCorrCoef(num_outputs=model.regressor.output_dim),
            ReducedKendallRankCorrCoef(num_outputs=model.regressor.output_dim),
            torchmetrics.R2Score(),
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanSquaredError(),
        ]
    )

    metrics = metrics.to(device)

    # Run a dummy forward pass to initialize Lazy layers
    dummy_batch = next(iter(train_loader))
    dummy_indices = dummy_batch["ids"].to(device)
    dummy_features = dummy_batch["features"].to(device)
    dummy_fractions = dummy_batch["fractions"].to(device)

    with torch.no_grad():
        _ = model(dummy_features, dummy_indices, dummy_fractions)


    mol_encoder_params = list(model.mol_encoder.parameters())
    other_params = [p for name, p in model.named_parameters() if not name.startswith("mol_encoder.")]

    optimizer = torch.optim.Adam([
        {
            'params': mol_encoder_params,
            'lr': lr_mol_encoder,
            'weight_decay': weight_decay,
        },
        {
            'params': other_params,
            'lr': lr_other,
            'weight_decay': weight_decay,
        }
    ])

    es = EarlyStopping(model, patience=patience, mode="minimize")

    # pbar = tqdm.tqdm(range(max_epochs))
    pbar = range(max_epochs)
    for epoch in pbar:
        model.train()

        overall_train_metrics = one_epoch(
            train_loader,
            optimizer,
            model,
            loss_fn,
            metrics,
            device,
            training=True,
        )

        metrics.reset()

        # validation + early stopping
        model.eval()
        with torch.no_grad():
            overall_val_metrics = one_epoch(
                val_loader,
                optimizer,
                model,
                loss_fn,
                metrics,
                device,
                training=False,
            )

        if wandb_logger:
            log_metrics(
                metrics=overall_train_metrics,
                epoch=epoch,
                logger=wandb_logger,
                mode="train",
            )
            log_metrics(
                metrics=overall_val_metrics,
                epoch=epoch,
                logger=wandb_logger,
                mode="val",
            )

        stop = es.check_criteria(overall_val_metrics["MeanAbsoluteError"], model)
        if stop:
            print(f"Early stop reached at {es.best_step} with {es.best_value}")
            break

        metrics.reset()

    # save model weights
    best_model_dict = es.restore_best()
    model.load_state_dict(best_model_dict)  # load the best one trained
    torch.save(model.state_dict(), f"{root_dir}/best_model_dict_{experiment_name}.pt")

    # final eval
    print("Using best model for a final eval")
    model.eval()
    with torch.no_grad():
        overall_val_metrics = one_epoch(
            val_loader,
            model,
            optimizer,
            loss_fn,
            metrics,
            device,
            training=False,
        )

    if wandb_logger:
        log_metrics(
            metrics=overall_val_metrics,
            epoch=epoch+1,
            logger=wandb_logger,
            mode="val",
        )

    if wandb_logger and wandb_logger == WandBLogger:
        wandb_logger.close()



def main(
    config,
    experiment_name,
    wandb_logger=None,
):
    config = copy.deepcopy(config)

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    print(f"Running on: {device}")

    root_dir = config.root_dir
    os.makedirs(root_dir, exist_ok=True)

    featurization = config.dataset.featurization

    if FEATURIZATION_TYPE[featurization] == "graphs" and config.mixture_model.mol_encoder.type != "gnn":
        raise ValueError(f"featurization is:{FEATURIZATION_TYPE[featurization]} but molecule encoder is: {config.mol_encoder.type}")

    if FEATURIZATION_TYPE[featurization] == "tensors" and config.mixture_model.mol_encoder.type == "gnn":
        raise ValueError(f"featurization is:{FEATURIZATION_TYPE[featurization]} but molecule encoder is: {config.mol_encoder.type}")

    # Dataset
    dataset = DATA_CATALOG[config.dataset.name]()
    task = config.dataset.task

    mixture_task = Mix2SmellData(
        dataset=dataset,
        task=task,
        featurization=featurization,
    )

    # Split Loader
    split_loader = SplitLoader(split_type="kfold")
    train_indices, val_indices, _ = split_loader(
        dataset_name= mixture_task.dataset.name,
        task=mixture_task.task,
        cache_dir=mixture_task.dataset.data_dir,
        split_num=0,
    )

    # Data Loader
    train_dataset = Subset(mixture_task, train_indices.tolist())
    val_dataset = Subset(mixture_task, val_indices.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=custom_collate,
        num_workers=config.num_workers,
    )

    model = build_mixture_model(config=config.mixture_model)
    model = model.to(device)

    # Save hyper parameters    
    OmegaConf.save(config, f"{root_dir}/hparams_{experiment_name}.yaml")

    # Training
    train(
        root_dir=root_dir,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_type=config.loss_type,
        lr_mol_encoder=config.lr_mol_encoder,
        lr_other=config.lr_other,
        device=device,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        patience=config.patience,
        experiment_name=experiment_name,
        wandb_logger=wandb_logger,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training and evaluation on a random split")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    parser.add_argument("--experiment_name", type=str, default="test_run", help="Name of the experiment")
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of the wandb project (optional)")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    experiment_name = args.experiment_name

    if args.wandb_project is not None:
        wandb_logger = WandBLogger(project=args.wandb_project)
    else:
        wandb_logger = None

    main(
        config=config,
        experiment_name=experiment_name,
        wandb_logger=wandb_logger,
    )
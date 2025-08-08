import enum
import sys
from typing import Optional

import torch
import torch.nn as nn
import torchmetrics
import tqdm
from torch.utils.data import DataLoader
from torchtune.training.metric_logging import WandBLogger

from mix2smell.model.utils import EarlyStopping
from mix2smell.model.custom_metrics import *

import os, random, numpy as np, torch


def log_metrics(metrics: torchmetrics.MetricCollection, epoch, logger, mode="train"):
    if isinstance(logger, WandBLogger):
        for metric in metrics.keys():
            logger.log(f"{mode}_{metric}", metrics[metric], epoch)
    else:
        logger.log({f"{mode}_{metric}": metrics[metric] for metric in metrics})


def compute_metrics(metrics: torchmetrics.MetricCollection, info):
    metric_dict = {k: item.cpu().item() for k, item in metrics.compute().items()}
    return info | metric_dict


def partial_z_score_normalization(pred, labels, mean, std):
    # apply z-score normalization to rata labels
    pred[:,2:] = (pred[:,2:] - mean)/(std + 1e-6)
    labels[:,2:] = (labels[:,2:] - mean)/(std + 1e-6)
    return pred, labels

def partial_min_max_normalization(
    pred: torch.Tensor,
    labels: torch.Tensor,
    mins: torch.Tensor,
    maxs: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Apply min-max normalization to only the RATA columns (i.e. cols 2:).
    preds[:,2:] and labels[:,2:] are scaled to [0,1] via (x - min)/(max - min).
    """
    denom = (maxs - mins) + eps
    pred[:, 2:]   = (pred[:, 2:]   - mins) / denom
    labels[:, 2:] = (labels[:, 2:] - mins) / denom
    return pred, labels



def one_epoch(
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    loss_fn,
    metrics: torchmetrics.MetricCollection,
    device: torch.device,
    training: bool = True,
    report_truncated_metrics: bool = True,
    return_predictions: bool = False,
    rata_mean=None,
    rata_std=None,
    rata_min=None,
    rata_max=None,
    normalize: bool = False,
    loss_type: str = "hurdle"
):
    running_loss = 0.0

    if training:
        model.train()
    else:
        model.eval()

    gather_preds = []
    gather_labels = []
    for i, batch in enumerate(loader):
        indices = batch["ids"].to(device)
        features = batch["features"].to(device)
        fractions = batch["fractions"].to(device)
        labels = batch["label"].to(device)

        if training:
            optimizer.zero_grad()

        out = model(features, indices, fractions)
        if isinstance(out, tuple):
            y_logits, y_pred = out
        else:
            y_logits = None
            y_pred = out

        if normalize:
            y_pred, labels = y_pred, labels

        if loss_type == "hurdle_gary":
            # use both heads for loss
            loss = loss_fn(y_logits, y_pred, labels)
            # mask predictions for metrics as in Gary's loop
            pos_mask = (labels > 0).float()
            gather_preds.append((y_pred * pos_mask).detach())
        else:
            loss = loss_fn(y_pred, labels)
            gather_preds.append(y_pred.detach())

        gather_labels.append(labels.detach())

        if training:
            l1_reg = 0.0
            l2_reg = 0.0
            for name, param in model.named_parameters(): # automate this with function so we can easily
                if 'weight' in name and param.requires_grad:
                    l1_reg += torch.norm(param, 1)
                    l2_reg += torch.norm(param, 2) ** 2

            l1_coeff = 1e-5
            l2_coeff = 0
            loss += l1_coeff * l1_reg + l2_coeff * l2_reg
            loss.backward()
            optimizer.step()

        running_loss += loss.detach().cpu().item()

    gather_preds = torch.concat(gather_preds)
    gather_labels = torch.concat(gather_labels)

    if report_truncated_metrics:
        metrics.update(gather_preds[:, 2:], gather_labels[:, 2:])
    else:
        metrics.update(gather_preds, gather_labels)

    overall_loss = running_loss / len(loader)
    overall_metrics = compute_metrics(metrics, {"loss": overall_loss})

    if return_predictions:
        return overall_metrics, gather_preds, gather_labels
    else:
        return overall_metrics




def init_metrics(data_loader, output_dim, device):
    # for rowwise on the full dataset
    n = len(data_loader.dataset)  

    metrics = torchmetrics.MetricCollection([
        ReducedPearsonCorrCoef(num_outputs=output_dim),
        RowwiseReducedPearsonCorrCoef(num_outputs=n),
        torchmetrics.CosineSimilarity(reduction='mean'),
        torchmetrics.R2Score(),
    ]).to(device)

    return metrics


def split_and_print(name, preds, trues, n_samples, output_dim, device):
    pi_m = torchmetrics.MetricCollection([
        ReducedPearsonCorrCoef(num_outputs=2),
        RowwiseReducedPearsonCorrCoef(num_outputs=n_samples),
        torchmetrics.CosineSimilarity(reduction='mean'),
        torchmetrics.R2Score(),
    ]).to(device)
    rata_m = torchmetrics.MetricCollection([
        ReducedPearsonCorrCoef(num_outputs=output_dim-2),
        RowwiseReducedPearsonCorrCoef(num_outputs=n_samples),
        torchmetrics.CosineSimilarity(reduction='mean'),
        torchmetrics.R2Score(),
    ]).to(device)

    pi_m.update(preds[:, :2].to(device), trues[:, :2].to(device))
    rata_m.update(preds[:, 2:].to(device), trues[:, 2:].to(device))

    pi = compute_metrics(pi_m, {})
    ra = compute_metrics(rata_m, {})

    print(
        f"  {name} PI   → Rowwise Pearson: {pi['RowwiseReducedPearsonCorrCoef']:.4f}, "
        f"Cosine: {pi['CosineSimilarity']:.4f}, R2: {pi['R2Score']:.4f}"
    )
    print(
        f"  {name} RATA → Rowwise Pearson: {ra['RowwiseReducedPearsonCorrCoef']:.4f}, "
        f"Cosine: {ra['CosineSimilarity']:.4f}, R2: {ra['R2Score']:.4f}"
    )


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    

# def train_one_epoch(
#     train_loader: DataLoader,
#     optimizer: torch.optim.Optimizer,
#     model: nn.Module,
#     loss_fn,
#     metrics: torchmetrics.MetricCollection,
#     device: torch.device,
# ):
#     train_loss = 0
#     num_batch = 0

#     # for i, batch in tqdm.tqdm(enumerate(train_loader)):
#     for i, batch in enumerate(train_loader):
#         train_indices = batch["ids"].to(device)
#         train_features = batch["features"].to(device)
#         train_fractions = batch["fractions"].to(device)
#         train_labels = batch["label"].to(device)

#         optimizer.zero_grad()

#         y_pred = model(train_features, train_indices, train_fractions)
#         loss = loss_fn(y_pred.view(-1), train_labels.view(-1))
#         # metrics.update(y_pred, train_labels)
#         metrics.update(y_pred.flatten(), train_labels.flatten())

#         loss.backward()
#         optimizer.step()

#         num_batch += i
#         train_loss += loss.detach().cpu().item()

#     # avg loss and metric per batch
#     overall_train_loss = train_loss / (num_batch + 1)
#     overall_train_metrics = compute_metrics(metrics, {"loss": overall_train_loss})

#     return overall_train_metrics


# def validate_one_epoch(
#     val_loader: DataLoader,
#     model: nn.Module,
#     loss_fn,
#     metrics: torchmetrics.MetricCollection,
#     device: torch.device,
# ):
#     val_loss = 0
#     num_batch = 0

#     # for i, batch in tqdm.tqdm(enumerate(val_loader)):
#     for i, batch in enumerate(val_loader):
#         val_indices = batch["ids"].to(device)
#         val_features = batch["features"].to(device)
#         val_fractions = batch["fractions"].to(device)
#         val_labels = batch["label"].to(device)

#         y_pred = model(val_features, val_indices, val_fractions)

#         loss = loss_fn(y_pred.view(-1), val_labels.view(-1))
#         # metrics.update(y_pred.flatten(), val_labels)
#         metrics.update(y_pred.flatten(), val_labels.flatten())

#         num_batch += i
#         val_loss += loss.detach().cpu().item()

#     # avg loss and metric per batch
#     overall_val_loss = val_loss / (num_batch + 1)
#     overall_val_metrics = compute_metrics(metrics, {"loss": overall_val_loss})

#     return overall_val_metrics


# def train(
#     root_dir: str,
#     model: nn.Module,
#     train_loader: DataLoader,
#     val_loader: DataLoader,
#     loss_type: str,
#     lr_mol_encoder: float,
#     lr_other: float,
#     device,
#     weight_decay: float,
#     max_epochs: int,
#     patience: Optional[int] = None,
#     experiment_name: Optional[str] = None,
#     wandb_logger: Optional[WandBLogger] = None,
# ):
#     loss_fn = LOSS_MAP[loss_type]()

#     metrics = torchmetrics.MetricCollection(
#         [
#             torchmetrics.PearsonCorrCoef(),
#             torchmetrics.R2Score(),
#             torchmetrics.MeanAbsoluteError(),
#             torchmetrics.MeanSquaredError(),
#             torchmetrics.KendallRankCorrCoef(),
#         ]
#     )

#     metrics = metrics.to(device)

#     # Run a dummy forward pass to initialize Lazy layers
#     dummy_batch = next(iter(train_loader))
#     dummy_indices = dummy_batch["ids"].to(device)
#     dummy_features = dummy_batch["features"].to(device)
#     dummy_fractions = dummy_batch["fractions"].to(device)

#     with torch.no_grad():
#         _ = model(dummy_features, dummy_indices, dummy_fractions)

#     # for name, param in model.named_parameters():
#     #     print(name, param.shape)

#     mol_encoder_params = list(model.mol_encoder.parameters())
#     other_params = [p for name, p in model.named_parameters() if not name.startswith("mol_encoder.")]

#     optimizer = torch.optim.Adam([
#         {
#             'params': mol_encoder_params,
#             'lr': lr_mol_encoder,
#             'weight_decay': weight_decay,
#         },
#         {
#             'params': other_params,
#             'lr': lr_other,
#             'weight_decay': weight_decay,
#         }
#     ])

#     es = EarlyStopping(model, patience=patience, mode="minimize")

#     # pbar = tqdm.tqdm(range(max_epochs))
#     pbar = range(max_epochs)
#     for epoch in pbar:
#         model.train()

#         overall_train_metrics = train_one_epoch(
#             train_loader,
#             optimizer,
#             model,
#             loss_fn,
#             metrics,
#             device,
#         )

#         metrics.reset()

#         # validation + early stopping
#         model.eval()
#         with torch.no_grad():
#             overall_val_metrics = validate_one_epoch(
#                 val_loader,
#                 model,
#                 loss_fn,
#                 metrics,
#                 device,
#             )

#         if wandb_logger:
#             log_metrics(
#                 metrics=overall_train_metrics,
#                 epoch=epoch,
#                 logger=wandb_logger,
#                 mode="train",
#             )
#             log_metrics(
#                 metrics=overall_val_metrics,
#                 epoch=epoch,
#                 logger=wandb_logger,
#                 mode="val",
#             )

#         # pbar.set_description(
#         #     f"Train: {overall_train_metrics['loss']:.4f} | Test: {overall_val_metrics['loss']:.4f} | Test pearson: {overall_val_metrics['PearsonCorrCoef']:.4f} | Test MAE: {overall_val_metrics['MeanAbsoluteError']:.4f}"
#         # )

#         stop = es.check_criteria(overall_val_metrics["MeanAbsoluteError"], model)
#         if stop:
#             print(f"Early stop reached at {es.best_step} with {es.best_value}")
#             break

#         metrics.reset()

#     # save model weights
#     best_model_dict = es.restore_best()
#     model.load_state_dict(best_model_dict)  # load the best one trained
#     torch.save(model.state_dict(), f"{root_dir}/best_model_dict_{experiment_name}.pt")

#     # final eval
#     print("Using best model for a final eval")
#     model.eval()
#     with torch.no_grad():
#         overall_val_metrics = validate_one_epoch(
#             val_loader,
#             model,
#             loss_fn,
#             metrics,
#             device,
#         )

#     if wandb_logger:
#         log_metrics(
#             metrics=overall_val_metrics,
#             epoch=epoch+1,
#             logger=wandb_logger,
#             mode="val",
#         )

#     if wandb_logger and wandb_logger == WandBLogger:
#         wandb_logger.close()
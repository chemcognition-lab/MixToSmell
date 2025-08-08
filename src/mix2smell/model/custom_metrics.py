import torch
from torchmetrics import Metric, PearsonCorrCoef, KendallRankCorrCoef
from torchmetrics.classification import MultilabelAUROC
from typing import Any, Dict, Optional

class SafeMultilabelAUROC(Metric):
    """
    A wrapper for MultilabelAUROC that safely casts the target tensor from
    float to long before updating the metric state.

    This is useful when ground truth labels are inadvertently loaded or
    processed as float tensors.
    """
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__()
        if 'num_outputs' in kwargs:
            kwargs['num_labels'] = kwargs['num_outputs']
            kwargs.pop('num_outputs')
        self.auroc = MultilabelAUROC(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Add a casting step to make it work with the original AUROC metric"""
        if target.is_floating_point():
            target = target.to(torch.long)
        
        self.auroc.update(preds, target)

    def compute(self) -> torch.Tensor:
        """Computes the final AUROC score."""
        return self.auroc.compute()

    def reset(self):
        """Resets the state of the internal metric."""
        self.auroc.reset()


class ReducedPearsonCorrCoef(Metric):
    """
    A wrapper for PearsonCorrCoef that allows for reducing the multi-output
    correlation vector into a single scalar value.

    Args:
        reduction (str, optional): The reduction method to apply to the
            per-output Pearson correlations. Can be 'mean', 'sum', 'max',
            'min', or 'none'. If 'none', no reduction is performed, and
            a tensor of correlations is returned. Defaults to 'mean'.
    """
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(self, reduction: Optional[str] = 'mean', **kwargs: Dict[str, Any]):
        super().__init__()

        if reduction is None:
            reduction = 'none' # Treat None as 'none'

        valid_reductions = ['mean', 'sum', 'max', 'min', 'none']
        if reduction not in valid_reductions:
            raise ValueError(
                f"Invalid reduction type: '{reduction}'. "
                f"Must be one of {valid_reductions}"
            )
        self.reduction = reduction

        # We create the "real" PearsonCorrCoef metric internally
        self.pearson = PearsonCorrCoef(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets."""
        self.pearson.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Computes the final Pearson Correlation and applies the specified reduction.
        """
        # First, compute the per-output correlations
        per_output_pearson = self.pearson.compute()

        # Then, apply the reduction based on the user's choice
        if self.reduction == 'mean':
            return per_output_pearson.mean()
        elif self.reduction == 'sum':
            return per_output_pearson.sum()
        elif self.reduction == 'max':
            return per_output_pearson.max()
        elif self.reduction == 'min':
            return per_output_pearson.min()
        else: # self.reduction == 'none'
            return per_output_pearson

    def reset(self):
        """Resets the state of the internal metric."""
        self.pearson.reset()


class RowwiseReducedPearsonCorrCoef(ReducedPearsonCorrCoef):
    """
    """
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(self, reduction: Optional[str] = 'mean', **kwargs: Dict[str, Any]):
        super().__init__(reduction, **kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets."""
        self.pearson.update(torch.t(preds), torch.t(target))



class ReducedKendallRankCorrCoef(Metric):
    """
    A wrapper for KendallRankCorrCoef that allows for reducing the multi-output
    correlation vector into a single scalar value.

    Args:
        reduction (str, optional): The reduction method to apply to the
            per-output Pearson correlations. Can be 'mean', 'sum', 'max',
            'min', or 'none'. If 'none', no reduction is performed, and
            a tensor of correlations is returned. Defaults to 'mean'.
    """
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(self, reduction: Optional[str] = 'mean', **kwargs: Dict[str, Any]):
        super().__init__()

        if reduction is None:
            reduction = 'none' # Treat None as 'none'

        valid_reductions = ['mean', 'sum', 'max', 'min', 'none']
        if reduction not in valid_reductions:
            raise ValueError(
                f"Invalid reduction type: '{reduction}'. "
                f"Must be one of {valid_reductions}"
            )
        self.reduction = reduction

        # We create the "real" KendallRankCorrCoef metric internally
        self.kendall = KendallRankCorrCoef(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets."""
        self.kendall.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Computes the final kendall Correlation and applies the specified reduction.
        """
        # First, compute the per-output correlations
        per_output_kendall = self.kendall.compute()

        # Then, apply the reduction based on the user's choice
        if self.reduction == 'mean':
            return per_output_kendall.mean()
        elif self.reduction == 'sum':
            return per_output_kendall.sum()
        elif self.reduction == 'max':
            return per_output_kendall.max()
        elif self.reduction == 'min':
            return per_output_kendall.min()
        else: # self.reduction == 'none'
            return per_output_kendall

    def reset(self):
        """Resets the state of the internal metric."""
        self.kendall.reset()
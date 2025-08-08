import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.version_info >= (3, 11):
    import enum
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

class TrainableTweedieLoss(nn.Module):
    """
    A learnable Tweedie loss with parameter p in (1,2) range.
    """
    def __init__(self, initial_p: float = 1.8):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(initial_p, dtype=torch.float32))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        mu = input.clamp(min=eps)
        y  = target.clamp(min=eps)
        p  = self.p.clamp(min=1.01, max=1.99)
        loss = y * mu.pow(1 - p) / (1 - p) - mu.pow(2 - p) / (2 - p)
        return loss.mean()


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE across P/I and RATA splits.
    Expects preds and targets with first 2 dims = P/I, rest = RATA.
    """
    def __init__(self, weight_pi: float = 1.0, weight_rata: float = 1.0):
        super().__init__()
        self.weight_pi = weight_pi
        self.weight_rata = weight_rata
        self.mse = nn.MSELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pi_pred, rata_pred = input[:, :2], input[:, 2:]
        pi_true, rata_true = target[:, :2], target[:, 2:]
        loss_pi   = self.mse(pi_pred, pi_true)
        loss_rata = self.mse(rata_pred, rata_true)
        return self.weight_pi * loss_pi + self.weight_rata * loss_rata


class HomoscedasticUncertaintyLoss(nn.Module):
    """
    Combines two losses with learned homoscedastic uncertainty weights:
      L = exp(-s1)*L1 + s1 + exp(-s2)*L2 + s2
    where s1, s2 are learnable log-variances.
    """
    def __init__(self):
        super().__init__()
        self.log_sigma1 = nn.Parameter(torch.zeros(()))
        self.log_sigma2 = nn.Parameter(torch.zeros(()))

    def forward(self, loss1: torch.Tensor, loss2: torch.Tensor) -> torch.Tensor:
        precision1 = torch.exp(-self.log_sigma1)
        precision2 = torch.exp(-self.log_sigma2)
        return (precision1 * loss1 + self.log_sigma1
              + precision2 * loss2 + self.log_sigma2)


class HuberLoss(nn.Module):
    """
    Wrapper around SmoothL1Loss (Huber) with configurable delta.
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(input, target, beta=self.delta)


class LogCosh(nn.Module):
    """
    Log-cosh loss: log(cosh(pred - true)), robust to outliers.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.cosh(input - target)).mean()


class QuantileLoss(nn.Module):
    """
    Pinball/Quantile loss for a given quantile tau (0<tau<1).
    """
    def __init__(self, tau: float = 0.5):
        super().__init__()
        assert 0 < tau < 1, "tau must be in (0,1)"
        self.tau = tau

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = target - input
        loss = torch.max(self.tau * diff, (self.tau - 1) * diff)
        return loss.mean()


class AsymMSE(nn.Module):
    """
    Column-wise weighted MSE: weights should broadcast to each column in true/pred.
    """
    def __init__(self):
        super().__init__()
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if weights is None:
            weights = torch.ones_like(target, device=target.device, dtype=target.dtype)

        return (weights * (input - target).pow(2)).mean()


class MixedTweedieHuberLoss(nn.Module):
    """
    Apply Tweedie loss on zeros and Huber on non-zeros per element.
    """
    def __init__(self, p: float = 1.8, delta: float = 1.0):
        super().__init__()
        self.tweedie = TrainableTweedieLoss(initial_p=p)
        self.huber   = HuberLoss(delta=delta)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        zero_mask = (target == 0)
        tw = self.tweedie(input[zero_mask], target[zero_mask]) if zero_mask.any() else 0.0
        hb = self.huber(input[~zero_mask], target[~zero_mask]) if (~zero_mask).any() else 0.0
        return tw + hb


class HurdleLoss(nn.Module):
    """
    A two‐part “hurdle” loss that
    1) learns a threshold τ to decide zero vs positive,
    2) applies BCE for the zero/nonzero mask,
    3) MSE on the positive portion only.
    """
    def __init__(self,
                 initial_tau: float = 0.1,
                 alpha: float = 10.0,
                 mask_weight: float = 1.0,
                 reg_weight: float = 1.0):
        super().__init__()
        # τ the cutoff, α the sharpness of the sigmoid
        self.tau = nn.Parameter(torch.tensor(initial_tau))
        self.alpha = alpha
        self.mask_weight = mask_weight
        self.reg_weight = reg_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, D) continuous predictions & true RATA intensities
        we’ll treat target>0 as “nonzero” mask
        """
        eps = 1e-6
        # 1) classification: mask_true is 1 if target > 0
        mask_true = (target > 0).float()
        # predicted logit for mask: we use pred itself relative to τ
        # pass through a scaled sigmoid to get p(nonzero)
        logit = self.alpha * (input - self.tau)
        p_nonzero = torch.sigmoid(logit)

        bce = F.binary_cross_entropy(p_nonzero.clamp(eps,1-eps), mask_true)

        # 2) regression MSE only on positives
        # mask the pred & target
        masked_pred = input * mask_true
        masked_tgt  = target * mask_true
        mse = F.mse_loss(masked_pred, masked_tgt)

        return self.mask_weight * bce + self.reg_weight * mse
    

class HurdleGaryLoss(nn.Module):
    """
    Gary's hurdle loss as a reusable nn.Module.
    Model must output two heads:
      - logits: raw scores for zero vs non-zero classification
      - preds: continuous predictions for regression

    Loss = BCEWithLogitsLoss(logits, mask) + MSE(preds*mask, target*mask) averaged over positives.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        preds: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # mask for positives
        mask = (target > 0).float()
        denom = mask.sum().clamp_min(1.0)  # avoid division by zero
        
        # classification loss: sum over all, then average by positive count 
        bce_sum = self.bce(logits, mask)
        bce = bce_sum / denom

        # regression loss: sum over positive entries, then average per positive
        reg_sum = F.mse_loss(preds * mask, target * mask, reduction='sum')
        reg = reg_sum / denom

        return bce + reg


# class CombinedRataLoss(nn.Module):
#     """
#     RATA‐only loss = α·BCE + β·RMSE + γ·(1 – Pearson) + δ·(1 – Cosine)
#                 + L1_residual + L2_residual
#     where α,β,γ,δ are learned parameters.
#     """
#     def __init__(
#         self,
#         initial_alpha: float = 1.0,
#         initial_beta:  float = 1.0,
#         initial_gamma: float = 1.0,
#         initial_delta: float = 1.0,
#         use_focal:     bool  = False,
#     ):
#         super().__init__()
#         # trainable combination weights
#         self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))
#         self.beta  = nn.Parameter(torch.tensor(float(initial_beta)))
#         self.gamma = nn.Parameter(torch.tensor(float(initial_gamma)))
#         self.delta = nn.Parameter(torch.tensor(float(initial_delta)))

#         self.use_focal = use_focal
#         if use_focal:
#             # you can pip-install “torchvision” or “timm” for a readily available FocalLoss
#             from torchvision.ops import sigmoid_focal_loss
#             self._focal = sigmoid_focal_loss
#         else:
#             self.bce = nn.BCEWithLogitsLoss()

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         eps = 1e-6
#         # -- 1) BCE on zero/nonzero mask --
#         mask_true = (target > 0).float()
#         if self.use_focal:
#             # focal takes logits, targets, alpha & gamma hyperparams
#             # here we just use defaults alpha=1, gamma=2
#             bce_loss = self._focal(input, mask_true, reduction="mean")
#         else:
#             bce_loss = self.bce(input, mask_true)

#         # -- 2) RMSE on raw intensities --
#         mse = F.mse_loss(input, target)
#         rmse = torch.sqrt(mse + eps)

#         # -- 3) Pearson correlation loss = 1 – corr(pred,true) --
#         p_mean = input.mean()
#         t_mean = target.mean()
#         cov    = ((input - p_mean) * (target - t_mean)).mean() # double check that this matches the pearson we're being evaluated on 
#         var_p  = ((input - p_mean).pow(2)).mean()
#         var_t  = ((target - t_mean).pow(2)).mean()
#         pearson = cov / (torch.sqrt(var_p * var_t) + eps)
#         pearson_loss = 1 - pearson

#         # -- 4) Cosine similarity loss = 1 – cos_sim(pred,true) --
#         cos_sim = F.cosine_similarity(input, target, dim=1).mean()
#         cos_loss = 1 - cos_sim

#         # -- 5) L1 and L2 on residuals --
#         l1 = F.l1_loss(input, target) # NEED TO DO WEIGHT_DECAY ON THE OPTIMIZER USE ADAMW
#         l2 = mse

#         # combine everything
#         return (
#             self.alpha * bce_loss
#           + self.beta  * rmse
#           + self.gamma * pearson_loss
#           + self.delta * cos_loss
#           + l1
#           + l2
#         )

class CombinedRataLoss(nn.Module):
    """
    RATA‐only loss = α·BCE + β·RMSE + γ·(1 – Pearson) + δ·(1 – Cosine)
                + L1_residual + L2_residual
    where α,β,γ,δ are learned parameters.
    """
    def __init__(
        self,
        initial_alpha: float = 1.0,
        initial_beta:  float = 1.0,
        initial_gamma: float = 1.0,
        initial_delta: float = 1.0,
        use_focal:     bool  = False,
    ):
        super().__init__()
        # trainable combination weights
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))
        self.beta  = nn.Parameter(torch.tensor(float(initial_beta)))
        self.gamma = nn.Parameter(torch.tensor(float(initial_gamma)))
        self.delta = nn.Parameter(torch.tensor(float(initial_delta)))

        self.use_focal = use_focal
        if use_focal:
            # you can pip-install “torchvision” or “timm” for a readily available FocalLoss
            from torchvision.ops import sigmoid_focal_loss
            self._focal = sigmoid_focal_loss
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        # -- 1) BCE on zero/nonzero mask --
        mask_true = (target > 0).float()
        if self.use_focal:
            # focal takes logits, targets, alpha & gamma hyperparams
            # here we just use defaults alpha=1, gamma=2
            bce_loss = self._focal(input, mask_true, reduction="mean")
        else:
            bce_loss = self.bce(input, mask_true)

        # -- 2) RMSE on raw intensities --
        mse = F.mse_loss(input, target)
        rmse = torch.sqrt(mse + eps)

        # -- 3) Pearson correlation loss = 1 – corr(pred,true) --
        y_pred_mu = input - torch.mean(input, dim=1, keepdim=True)
        y_true_mu = target - torch.mean(target, dim=1, keepdim=True)
        cov = torch.sum(y_pred_mu * y_true_mu, dim=1)
        y_pred_std = torch.sqrt(torch.sum(y_pred_mu ** 2, dim=1) + eps)
        y_true_std = torch.sqrt(torch.sum(y_true_mu ** 2, dim=1) + eps)
        pearson = cov / (y_pred_std * y_true_std)
        pearson_loss = 1 - pearson
        pearson_loss = torch.mean(pearson_loss)

        # -- 4) Cosine similarity loss = 1 – cos_sim(pred,true) --
        cos_sim = F.cosine_similarity(input, target, dim=1).mean()
        cos_loss = 1 - cos_sim

        # -- 5) L1 and L2 on residuals --
        l1 = 0 #F.l1_loss(input, target) # NEED TO DO WEIGHT_DECAY ON THE OPTIMIZER USE ADAMW
        l2 = 0 #mse

        # combine everything
        return (
            self.alpha * bce_loss
          + self.beta  * rmse
          + self.gamma * pearson_loss
          + self.delta * cos_loss
          + l1
          + l2
        )    


class CombinedRataLossUncertainty(nn.Module):
    """
    RATA-only composite loss with learned uncertainty weights.
    
    Loss = Σ_i [ exp(-s_i) * L_i + s_i ]
    
    where s_i = log variance for sub-loss i.
    """
    def __init__(self, use_focal: bool = False):
        super().__init__()
        # one s_i per sub-loss: BCE, RMSE, Pearson, Cosine
        self.log_var_bce     = nn.Parameter(torch.zeros(1))
        self.log_var_rmse   = nn.Parameter(torch.zeros(1))
        self.log_var_pear   = nn.Parameter(torch.zeros(1))
        self.log_var_cos    = nn.Parameter(torch.zeros(1))

        self.use_focal = use_focal
        if use_focal:
            from torchvision.ops import sigmoid_focal_loss
            self._focal = sigmoid_focal_loss
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6

        # -- 1) BCE on mask --
        mask = (target > 0).float()
        if self.use_focal:
            bce = self._focal(input, mask, reduction="mean")
        else:
            bce = self.bce(input, mask)

        # -- 2) RMSE on intensities --
        mse = F.mse_loss(input, target, reduction="mean")
        rmse = torch.sqrt(mse + eps)

        # -- 3) Pearson loss = 1 - corr --
        p_mean = input.mean(dim=0)
        t_mean = target.mean(dim=0)
        cov    = ((input - p_mean) * (target - t_mean)).mean()
        var_p  = ((input - p_mean).pow(2)).mean()
        var_t  = ((target - t_mean).pow(2)).mean()
        pear = cov / (torch.sqrt(var_p * var_t) + eps)
        pearson_loss = 1 - pear

        # -- 4) Cosine loss = 1 - mean cos_sim --
        cos_sim   = F.cosine_similarity(input, target, dim=1).mean()
        cos_loss  = 1 - cos_sim

        # -- combine with uncertainty weights --
        loss = (
            bce    * torch.exp(-self.log_var_bce)   + self.log_var_bce   +
            rmse   * torch.exp(-self.log_var_rmse) + self.log_var_rmse +
            pearson_loss * torch.exp(-self.log_var_pear)  + self.log_var_pear +
            cos_loss     * torch.exp(-self.log_var_cos)   + self.log_var_cos
        )

        return loss


class LossEnum(StrEnum):
    """Basic str enum for molecule aggregators."""

    mae = enum.auto()
    mse = enum.auto()
    bce = enum.auto()
    tweedie = enum.auto()
    weighted_mse = enum.auto()
    uncertainty = enum.auto()
    huber = enum.auto()
    logcosh = enum.auto()
    quantile = enum.auto()
    asym_mse = enum.auto()
    mixed = enum.auto()
    hurdle = enum.auto()
    hurdle_gary = enum.auto()
    combined_rata = enum.auto()
    combined_rata_uncertainty = enum.auto()


LOSS_MAP = {
    LossEnum.mae: nn.L1Loss,
    LossEnum.mse: nn.MSELoss,
    LossEnum.bce: nn.BCEWithLogitsLoss,
    LossEnum.tweedie: TrainableTweedieLoss,
    LossEnum.weighted_mse: lambda args: WeightedMSELoss(args.weight_pi, args.weight_rata),
    LossEnum.uncertainty: HomoscedasticUncertaintyLoss,
    LossEnum.hurdle: lambda args: HurdleLoss(initial_tau=float(args.initial_tau),
                                              alpha=float(args.alpha),
                                              mask_weight=float(args.mask_weight),
                                              reg_weight=float(args.reg_weight)),
    LossEnum.hurdle_gary: lambda args: HurdleGaryLoss(),
    LossEnum.huber: lambda args: HuberLoss(delta=args.huber_delta),
    LossEnum.logcosh: LogCosh,
    LossEnum.quantile: lambda args: QuantileLoss(tau=args.quantile_tau),
    LossEnum.asym_mse: AsymMSE,
    LossEnum.mixed: lambda args: MixedTweedieHuberLoss(delta=args.huber_delta),
    LossEnum.combined_rata: lambda args: CombinedRataLoss(
        initial_alpha=args.initial_alpha,
        initial_beta=args.initial_beta,
        initial_gamma=args.initial_gamma,
        initial_delta=args.initial_delta,
        use_focal=args.use_focal,
    ),
    LossEnum.combined_rata_uncertainty: lambda args: CombinedRataLossUncertainty(
        use_focal=args.use_focal,
    )
}

def get_loss_fn(loss_type, loss_args, device):
    """
    Instantiate and return:
      - loss_fn: either a 2-arg function (preds, true) or, for hurdle_gary,
                 the raw HurdleGaryLoss module (logits, preds, true).
      - combined_mod: an optional nn.Module for extra trainable params.
    """
    import torch.nn as _nn

    # 1) build the underlying loss object
    loss_obj = LOSS_MAP[loss_type](loss_args)

    # if it’s a module, move it to the device
    if isinstance(loss_obj, _nn.Module):
        loss_obj = loss_obj.to(device)

    combined_mod = None
    if loss_type == "combined_rata":
        combined_mod = loss_obj
    elif loss_type == "combined_rata_uncertainty":
        combined_mod = loss_obj

    # 2) special‐case hurdle_gary: use the module itself
    if loss_type == "hurdle_gary":
        # loss_obj is HurdleGaryLoss; it implements forward(logits, preds, target)
        return loss_obj, combined_mod

    # 3) otherwise wrap into the standard (preds, true) API
    def loss_fn(preds: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # split out PI vs. RATA
        pi_pred, rata_pred = preds[:, :2], preds[:, 2:]
        pi_true, rata_true = true[:, :2], true[:, 2:]

        # PI-MSE
        L_pi = LOSS_MAP["mse"]()(pi_pred, pi_true).to(device)
        # RATA via our loss_obj
        L_r  = loss_obj(rata_pred, rata_true).to(device)

        # weighted‐MSE special
        if loss_type == "weighted_mse":
            L_r -= loss_args.weight_pi * L_pi

        # uncertainty‐wrapper
        if getattr(loss_args, "use_uncertainty", False):
            return LOSS_MAP["uncertainty"]()(L_pi, L_r)

        # otherwise sum with hyper‐weights
        if loss_type != "weighted_mse":
            return loss_args.weight_pi * L_pi + loss_args.weight_rata * L_r

        # fallback
        return L_r + loss_args.weight_pi * L_pi

    return loss_fn, combined_mod
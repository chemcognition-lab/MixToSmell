import torch
import torch.nn as nn

from mix2smell.model.utils import ActivationEnum, ACTIVATION_MAP

R = 8.63e-5

class HurdleHead(torch.nn.Module):
    def __init__(
        self, 
        hidden_dim: int = 100,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        if num_layers == 0:
            self.layers = nn.Identity()
        else:
            self.layers = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ReLU())
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.BatchNorm1d(hidden_dim))
                self.layers.append(nn.Dropout(p=dropout_rate))
        
        self.classifier = torch.nn.LazyLinear(output_dim)
        self.regressor = torch.nn.Sequential(
            torch.nn.LazyLinear(output_dim),
            torch.nn.Softplus()     # smoothly enforces positive values
        )

    def forward(self, x):
        h = self.layers(x)
        return self.classifier(h), self.regressor(h)


class PredictiveHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 100,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout_rate: float = 0.1,
    ):
        super(PredictiveHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        if num_layers == 0:
            self.layers = nn.Sequential(nn.LazyLinear(output_dim))
        else:
            self.layers = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ReLU())
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.BatchNorm1d(hidden_dim))
                self.layers.append(nn.Dropout(p=dropout_rate))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        output = self.layers(x)
        return output


def rk_mixing_law():
    # TODO
    return 1


def vft_conductivity_cmu(c1, c2, c3, temp):
    cond = c1*torch.exp(-c2/(temp - c3))
    return cond.unsqueeze(1)


def arr_conductivity(logA, Ea, temp):
    """
    Computes conductivity from Arrhenius fit parameters.

    :param logA: logA value for fit.
    :param Ea: Ea value for fit.
    :param temp: temperature values for each point.
    :return: ionic conductivity calculation.
    """

    e = torch.exp(torch.tensor([1], device=logA.device))
    C = torch.log10(e)/R
    
    cond = logA - C*Ea/temp

    return cond.unsqueeze(1)


def vft_conductivity_mit(logA, Ea, T0, temp):
    """
    Computes conductivity from Arrhenius fit parameters.

    :param logA: logA value for fit.
    :param Ea: Ea value for fit.
    :param T0: T0 fit value.
    :param temp: temperature values for each point.
    :return: ionic conductivity calculation.
    """

    e = torch.exp(torch.tensor([1], device=logA.device))
    C = torch.log10(e)/R
    
    cond = logA - C*Ea/(temp - T0)

    return cond.unsqueeze(1)


def fits_to_conds(preds, temps, law):
    """
    Computes conductivity from Arrhenius or vtf fit parameters.

    :param preds: parameters outputs of model.
    :param temps: temperature values for each point.
    :param arr_vtf: arrhenius or vtf parameters
    :return: ionic conductivity values.
    """
    
    param1 = preds[:,0]
    param2 = preds[:,1]
    if "vft" in law:
        param3 = preds[:,2]

    temps = temps.view(-1)
        
    #run fit parameters with temps through equations to get conductivity values         
    if law == "arrhenius":
        conds = arr_conductivity(logA=param1, Ea=param2, temp=temps)
    elif law == "vft_mit":
        conds = vft_conductivity_mit(logA=param1, Ea=param2, T0=param3, temp=temps)
    elif law == "vft_cmu":
        conds = vft_conductivity_cmu(c1=param1, c2=param2, c3=param3, temp=temps)
        
    return conds


LAW_TO_NUM_PARAMS = {
    "vft_cmu": 3,
    "vft_mit": 3,
    "arrhenius": 2,
}


class PhysicsPredictiveHead(nn.Module):
    def __init__(
        self,
        law: str,
        hidden_dim: int = 100,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super(PhysicsPredictiveHead, self).__init__()

        if law not in LAW_TO_NUM_PARAMS:
            raise ValueError(f"Law '{law}' is not recognized. Choose from {list(LAW_TO_NUM_PARAMS.keys())}.")

        self.law = law
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        output_dim = LAW_TO_NUM_PARAMS[law]

        self.layers = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(p=dropout_rate))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x, x_temperatures):
        preds = self.layers(x)
        output = fits_to_conds(preds=preds, temps=x_temperatures, law=self.law)
        return output


class CosineRegressor(nn.Module):
    """Cosine similarity as regressor, is PI."""

    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        x1, x2 = x.unbind(dim=-1)
        sim = self.cosine_similarity(x1, x2).unsqueeze(-1)
        return 1.0 - sim


class ScaledCosineRegressor(nn.Module):
    """Use scaled cosine similarity as similarity regressor, is PI.

    We add a scaling layer since we have observed that cosine similarity
    will not always match the target range of the similarity score.
    Sigmoid is to restrict the output to [0, 1].
    """

    def __init__(self, output_dim: int, act: ActivationEnum, no_bias: bool = False):
        super().__init__()
        self.cosine_distance = CosineRegressor()
        self.scaler = nn.Linear(output_dim, output_dim, bias=not no_bias)
        self.activation = ACTIVATION_MAP[act]()

        # clamp the scalar, this is required since our distance metric is only defined in this range
        with torch.no_grad():
            self.scaler.weight.clamp_(min=0)
            for p in self.scaler.parameters():
                p.copy_(nn.Parameter(torch.ones_like(p) * 0.5))

    def forward(self, x):
        cos_dist = self.cosine_distance(x)
        return self.activation(self.scaler(cos_dist))

class AdaptiveOutputLayer(torch.nn.Module):
    """
    Adaptive output layer for transfer learning between datasets with different output dimensions.
    Expands the output layer to match the new dataset, optionally freezing old weights.
    """
    def __init__(
            self,
            encoder_output_dim: int,
            old_head_input_dim: int,
            old_output_dim: int,
            new_output_dim: int,
            freeze_old: bool = True,
    ):    
        super().__init__()
        self.old_output_dim = old_output_dim
        self.new_output_dim = new_output_dim
        self.encoder_output_dim = encoder_output_dim
        self.freeze_old = freeze_old

        # Use explicit Linear for adaptive_layer, LazyLinear for old_head
        self.adaptive_layer = nn.Linear(encoder_output_dim, old_head_input_dim)
        self.old_head = nn.Linear(old_head_input_dim, old_output_dim)
        self.new_head = nn.Linear(old_head_input_dim, new_output_dim)

        if freeze_old:
            for param in self.old_head.parameters():
                param.requires_grad = False

    def forward(self, x):
        # 1) Project into the old head input space
        x_proj = self.adaptive_layer(x)    # -> [batch, old_head_input_dim]

        # 2) Optionally run the pretrained old head
        # baseline = self.old_head(x_proj)  # -> [batch, old_output_dim]

        # 3 Run new head for the new dataset (Keller)
        new_out = self.new_head(x_proj)     # -> [batch, new_output_dim]
        return new_out                     # now [batch, new_output_dim]

class TwoHeadRegressor(nn.Module):
    """
    A “P/I head” for the first 2 outputs and a “RATA head” for the remaining 51,
    then concatenates them back together.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        # first 2 columns → P/I
        self.pi_head = PredictiveHead(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=2,
            dropout_rate=dropout_rate,
        )
        # remaining columns → RATA
        self.rata_head = PredictiveHead(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim - 2,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        pi   = self.pi_head(x)    # shape (B,2)
        rata = self.rata_head(x)  # shape (B, total_output_dim-2)
        return torch.cat([pi, rata], dim=-1)
import numpy as np
from typing import Callable, Dict, List, Union
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat import quat_scale_to_covar_preci
from gsplat.relocation import compute_relocation
from gsplat.utils import normalized_quat_to_rotmat, xyz_to_polar


@torch.no_grad()
def _multinomial_sample(weights: Tensor, n: int, replacement: bool = True) -> Tensor:
    """Sample from a distribution using torch.multinomial or numpy.random.choice.

    This function adaptively chooses between `torch.multinomial` and `numpy.random.choice`
    based on the number of elements in `weights`. If the number of elements exceeds
    the torch.multinomial limit (2^24), it falls back to using `numpy.random.choice`.

    Args:
        weights (Tensor): A 1D tensor of weights for each element.
        n (int): The number of samples to draw.
        replacement (bool): Whether to sample with replacement. Default is True.

    Returns:
        Tensor: A 1D tensor of sampled indices.
    """
    num_elements = weights.size(0)

    if num_elements <= 2**24:
        # Use torch.multinomial for elements within the limit
        return torch.multinomial(weights, n, replacement=replacement)
    else:
        # Fallback to numpy.random.choice for larger element spaces
        weights = weights / weights.sum()
        weights_np = weights.detach().cpu().numpy()
        sampled_idxs_np = np.random.choice(
            num_elements, size=n, p=weights_np, replace=replacement
        )
        sampled_idxs = torch.from_numpy(sampled_idxs_np)

        # Return the sampled indices on the original device
        return sampled_idxs.to(weights.device)


@torch.no_grad()
def _update_param_with_optimizer(
    param_fn: Callable[[str, Tensor], Tensor],
    optimizer_fn: Callable[[str, Tensor], Tensor],
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    names: Union[List[str], None] = None,
):
    """Update the parameters and the state in the optimizers with defined functions.

    Args:
        param_fn: A function that takes the name of the parameter and the parameter itself,
            and returns the new parameter.
        optimizer_fn: A function that takes the key of the optimizer state and the state value,
            and returns the new state value.
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        names: A list of key names to update. If None, update all. Default: None.
    """
    if names is None:
        # If names is not provided, update all parameters
        names = list(params.keys())

    for name in names:
        param = params[name]
        new_param = param_fn(name, param)
        params[name] = new_param
        if name not in optimizers:
            assert not param.requires_grad, (
                f"Optimizer for {name} is not found, but the parameter is trainable."
                f"Got requires_grad={param.requires_grad}"
            )
            continue
        optimizer = optimizers[name]
        for i in range(len(optimizer.param_groups)):
            param_state = optimizer.state[param]
            del optimizer.state[param]
            for key in param_state.keys():
                if key != "step":
                    v = param_state[key]
                    param_state[key] = optimizer_fn(key, v)
            optimizer.param_groups[i]["params"] = [new_param]
            optimizer.state[new_param] = param_state


@torch.no_grad()
def duplicate(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    """Inplace duplicate the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to duplicate the Gaussians.
    """
    device = mask.device
    sel = torch.where(mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(torch.cat([p, p[sel]]), requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))


@torch.no_grad()
def split(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        mask: Tensor,
        revised_opacity: bool = False,
):
    """Inplace split the Gaussians with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: A dictionary of extra running states (Tensors).
        mask: A boolean mask (True => the Gaussians to be split).
        revised_opacity: Whether to use the revised opacity formulation
            from arXiv:2404.06109. Default: False.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]

    w_inv = 1 / torch.exp(params["w"][sel]).unsqueeze(1)

    means = params["means"][sel]
    scales = (
            torch.exp(params["scales"][sel])
            * w_inv
            * torch.norm(means, dim=1).unsqueeze(1)
    )
    quats = F.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)

    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales,
        torch.randn(2, len(scales), 3, device=device),
    )

    new_means = (means * w_inv + samples).reshape(-1, 3)
    _, new_w, _ = xyz_to_polar(new_means)
    new_means = new_means * new_w.unsqueeze(1)

    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            p_split = new_means
        elif name == "scales":
            p_split = torch.log(torch.exp(params["scales"][sel]) / 1.6).repeat(2, 1)
        elif name == "opacities" and revised_opacity:
            new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
            p_split = torch.logit(new_opacities).repeat(repeats)
        elif name == "w":
            p_split = torch.log(new_w)
        else:
            p_split = p[sel].repeat(repeats)
        p_new = torch.cat([p[rest], p_split])
        p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        return p_new

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
        return torch.cat([v[rest], v_split])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            state[k] = torch.cat((v[rest], v_new))


@torch.no_grad()
def split_edc(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    revised_opacity: bool = False,
):
    """Inplace split the Gaussian with the given mask, adapted to mimic densify_and_split_EDC.

    Args:
        params: A dictionary of parameters (e.g., "means", "scales", "opacities", "quats", "w").
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: A dictionary of extra state tensors.
        mask: A boolean mask to select Gaussians for splitting.
        revised_opacity: Whether to use the revised opacity formulation (arXiv:2404.06109).
                         If False, uses EDC-like opacity reduction. Default: False.
    """
    device = mask.device
    sel = torch.where(mask)[0]  # Indices of Gaussians to split
    rest = torch.where(~mask)[0]  # Indices of Gaussians to keep unchanged

    if len(sel) == 0:
        return  # No Gaussians to split

    # w_inv is used to convert from homogeneous to standard 3D coords.
    w_inv = 1 / torch.exp(params["w"][sel]).unsqueeze(1)  # [N, 1]
    means = params["means"][sel]  # [N, 3] in homogeneous representation
    # Convert scales to real 3D scales for splitting calculations
    scales_log = params["scales"][sel]  # [N, 3] (log scale)
    scales_3d = (
        torch.exp(scales_log) * w_inv * torch.norm(means, dim=1, keepdim=True)
    )  # [N, 3] in real 3D

    # Compute dominant axis using these real 3D scales
    max_indices = torch.argmax(scales_3d, dim=1)  # [N]
    offset_magnitudes = 1.5 * scales_3d[torch.arange(len(sel)), max_indices]  # [N]

    # Prepare offsets in local space, along the dominant axis
    signs = torch.tensor([1.0, -1.0], device=device).view(2, 1)  # [2, 1]
    offsets_local = torch.zeros(2, len(sel), 3, device=device)  # [2, N, 3]
    offsets_local[
        torch.arange(2)[:, None],  # [0,1] in dim 0
        torch.arange(len(sel)),    # [0..N-1] in dim 1
        max_indices
    ] = signs * offset_magnitudes

    # Rotate offsets to world space
    quats = F.normalize(params["quats"][sel], dim=-1)  # [N, 4]
    rotmats = normalized_quat_to_rotmat(quats)         # [N, 3, 3]
    offsets_world = torch.einsum("nij,bnj->bni", rotmats, offsets_local)  # [2, N, 3]

    # Convert means from homogeneous to 3D, apply offset, then reshape
    means_3d = means * w_inv  # [N, 3] in standard 3D
    new_means_3d = (means_3d.unsqueeze(0) + offsets_world).reshape(-1, 3)  # [2N, 3]

    # Get new radial distance (w in polar sense), and re-embed in homogeneous coords
    _, new_w, _ = xyz_to_polar(new_means_3d)  # new_w: [2N]
    new_means = new_means_3d * new_w.unsqueeze(1)  # re-embed: [2N, 3]

    # Adjust scaling log-values: shrink on dominant axis, mild shrink on others
    log_0_5 = math.log(0.5)
    log_0_85 = math.log(0.85)
    adj = torch.full((len(sel), 3), log_0_85, device=device)
    adj[torch.arange(len(sel)), max_indices] = log_0_5
    new_scales_log = (scales_log + adj).repeat(2, 1)  # [2N, 3]

    def param_fn(name: str, p: Tensor) -> Tensor:
        """Update parameter tensors with split values."""
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            # Use the newly computed homogeneous means
            p_split = new_means
        elif name == "scales":
            # Use the newly computed log scales
            p_split = new_scales_log
        elif name == "w":
            # w is log of the radial distance
            p_split = torch.log(new_w)  # [2N]
            if p.dim() > 1:  # if shape is [N, 1], repeat accordingly
                p_split = p_split.unsqueeze(1)
        elif name == "opacities":
            opacity = torch.sigmoid(p[sel])
            if revised_opacity:
                new_opacity = 1.0 - torch.sqrt(1.0 - opacity)
            else:
                new_opacity = 0.6 * opacity
            p_split_logit = torch.logit(new_opacity)
            if p.dim() == 1:
                p_split = p_split_logit.repeat(2)
            else:
                p_split = p_split_logit.repeat(2, 1)
        else:
            # Default: just repeat the original
            p_split = p[sel].repeat(repeats)

        p_new = torch.cat([p[rest], p_split], dim=0)
        p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        return p_new

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        """Update optimizer state."""
        v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
        return torch.cat([v[rest], v_split], dim=0)

    # Update parameters and optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    # Update state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            state[k] = torch.cat((v[rest], v_new), dim=0)


@torch.no_grad()
def remove(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    """Inplace remove the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to remove the Gaussians.
    """
    sel = torch.where(~mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(p[sel], requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return v[sel]

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v[sel]


@torch.no_grad()
def reset_opa(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    value: float,
):
    """Inplace reset the opacities to the given post-sigmoid value.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        value: The value to reset the opacities
    """

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            opacities = torch.clamp(p, max=torch.logit(torch.tensor(value)).item())
            return torch.nn.Parameter(opacities, requires_grad=p.requires_grad)
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )


@torch.no_grad()
def relocate(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    binoms: Tensor,
    min_opacity: float = 0.005,
):
    """Inplace relocate some dead Gaussians to the lives ones.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to indicates which Gaussians are dead.
    """
    # support "opacities" with shape [N,] or [N, 1]
    opacities = torch.sigmoid(params["opacities"])

    dead_indices = mask.nonzero(as_tuple=True)[0]
    alive_indices = (~mask).nonzero(as_tuple=True)[0]
    n = len(dead_indices)

    # Sample for new GSs
    eps = torch.finfo(torch.float32).eps
    probs = opacities[alive_indices].flatten()  # ensure its shape is [N,]
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    sampled_idxs = alive_indices[sampled_idxs]
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p[dead_indices] = p[sampled_idxs]
        return torch.nn.Parameter(p, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v[sampled_idxs] = 0
        return v

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v[sampled_idxs] = 0


@torch.no_grad()
def sample_add(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    n: int,
    binoms: Tensor,
    min_opacity: float = 0.005,
):
    opacities = torch.sigmoid(params["opacities"])

    eps = torch.finfo(torch.float32).eps
    probs = opacities.flatten()
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p_new = torch.cat([p, p[sampled_idxs]])
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
        return torch.cat([v, v_new])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v_new))


@torch.no_grad()
def inject_noise_to_position(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    scaler: float,
):
    opacities = torch.sigmoid(params["opacities"].flatten())
    scales = torch.exp(params["scales"])
    covars, _ = quat_scale_to_covar_preci(
        params["quats"],
        scales,
        compute_covar=True,
        compute_preci=False,
        triu=False,
    )

    def op_sigmoid(x, k=100, x0=0.995):
        return 1 / (1 + torch.exp(-k * (x - x0)))

    noise = (
        torch.randn_like(params["means"])
        * (op_sigmoid(1 - opacities)).unsqueeze(-1)
        * scaler
    )
    noise = torch.einsum("bij,bj->bi", covars, noise)
    params["means"].add_(noise)
    
@torch.no_grad()
def inject_noise_to_position_new(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, torch.Tensor],
    scaler: float,
):
    """
    Inject random noise into 'means' weighted by (1 - alpha)^100, 
    then transform by the local covariance to preserve orientation/scale.

    :param params: Dict of parameter tensors (requires ["means", "opacities", "scales", "quats"]).
    :param optimizers: Dict of optimizers (not actually modified here).
    :param state: Custom dictionary state (not used here).
    :param scaler: Base scaling factor for noise.
    """
    # 1) Compute alpha and convert to weighting = (1 - alpha)^100
    opacities = torch.sigmoid(params["opacities"].flatten())  # shape [N]
    alpha_weight = (1.0 - opacities).pow(100)
    w_inv = 1 / torch.exp(params["w"]).unsqueeze(1)

    means = params["means"]
    scales = (
            torch.exp(params["scales"])
            * w_inv
            * torch.norm(means, dim=1).unsqueeze(1)
    )

    # 2) Compute local covariance for each Gaussian (orientation & scale)
    covars, _ = quat_scale_to_covar_preci(
        params["quats"],
        scales,
        compute_covar=True,
        compute_preci=False,
        triu=False,
    )

    # 3) Create noise ~ N(0,1), scale by alpha_weight and user 'scaler'
    noise = (
        torch.randn_like(params["means"])
        * alpha_weight.unsqueeze(-1)
        * scaler
    )

    # 4) Transform noise by the covariance so it respects each Gaussian's orientation
    #    covars is shape [N,3,3], noise is [N,3]
    noise = torch.einsum("bij,bj->bi", covars, noise)
    cart_new = means * w_inv + noise

    _, r_new, _ = xyz_to_polar(cart_new)
    means_new = cart_new * r_new.unsqueeze(1)

    # Store
    params["means"].data = means_new
    params["w"].data = torch.log(r_new.clamp_min(1e-8))

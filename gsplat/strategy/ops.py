import numpy as np
from typing import Callable, Dict, List, Union
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat import quat_scale_to_covar_preci
from gsplat.relocation import compute_relocation
from gsplat.utils import normalized_quat_to_rotmat


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

# BOGausS: Better Optimized Gaussian Splatting
# https://arxiv.org/abs/2504.01844
@torch.no_grad()
def split_new(
        params: Dict[str, torch.nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        revised_opacity: bool = False,
        alpha_t: float = 1.0,
        alpha_g: float = 0.2,
):
    """
    Split each selected Gaussian (where `mask[i] == True`) into 2 children.

    This version:
      • Uses *homogeneous coordinates* by referencing a parameter "w".
      • Recomputes new means from polar coords: xyz -> (r, w, phi), etc.
      • Logs and reassigns "w" so that new children get a consistent homogeneous coordinate.
      • Partially inherits optimizer states for "exp_avg" and "exp_avg_sq", etc.
      • Leaves ephemeral `state[...]` buffers with 2 new zero rows for each splitted Gaussian.

    Args:
      params: Dictionary of all model parameters, including "means", "scales", "quats", "opacities", and "w".
      optimizers: Dictionary of corresponding optimizers (e.g. Adam).
      state: Extra running states (like "grad2d", "count", etc.).
      mask: Boolean mask, shape [N], indicating which Gaussians to split.
      revised_opacity: If True, apply revised opacity from arXiv:2404.06109.
      alpha_t, alpha_g: partial-inheritance factors for the optimizer's Adam states.
    """

    device = mask.device
    sel = torch.where(mask)[0]  # indices of the "mother" Gaussians
    rest = torch.where(~mask)[0]  # indices of the remaining (father) Gaussians
    N_sel = len(sel)
    if N_sel == 0:
        return  # nothing to split

    old_count = len(rest) + len(sel)  # total before splitting

    mother_means = params["means"][sel]  # shape [N_sel, 3]
    # scales = exp(params["scales"]) * w_inv * ||means||
    # the old code does this:
    r_means = torch.norm(mother_means, dim=1).unsqueeze(1)  # shape [N_sel, 1]
    mother_scales = torch.exp(params["scales"][sel])
    # Rotation from quats:
    mother_quats = F.normalize(params["quats"][sel], dim=-1)  # shape [N_sel, 4]
    rotmats = normalized_quat_to_rotmat(mother_quats)  # shape [N_sel, 3,3]

    # sample 2 random offsets per mother, shape [2, N_sel, 3]
    rand_samples = torch.randn(2, N_sel, 3, device=device)
    # local_shifts = R * scales * random
    local_shifts = torch.einsum("nij,nj,bnj->bni", rotmats, mother_scales, rand_samples)

    # new_means in "Cartesian coords" = (mother_means * w_inv + local_shifts)
    # shape => [2, N_sel, 3] => flatten => [2*N_sel, 3]
    child_means_cart = (mother_means + local_shifts).reshape(-1, 3)

    ####################################################
    # 2) Build param_fn for all param keys
    ####################################################
    def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
        """
        For each param name, produce the new [new_count, ...] parameter data,
        reusing the "father" rows from `rest` plus new splitted rows for `sel`.
        """
        if p.shape[0] != old_count:
            # not a per-Gaussian param, skip
            return p

        # father part
        father_part = p[rest]
        mother_part = p[sel]

        if name == "means":
            # We just computed 'child_means' above, shape [2*N_sel, 3]
            splitted = child_means_cart
            new_p = torch.cat([father_part, splitted], dim=0)
            return torch.nn.Parameter(new_p, requires_grad=p.requires_grad)

        elif name == "scales":
            # replicate the old logic: new scales = log(scales/1.6)
            splitted = torch.log(torch.exp(params["scales"][sel]) / 1.6).repeat(2, 1)
            new_p = torch.cat([father_part, splitted], dim=0)
            return torch.nn.Parameter(new_p, requires_grad=p.requires_grad)

        elif name == "quats":
            # replicate mother quats 2x
            splitted = mother_part.repeat(2, 1)
            new_p = torch.cat([father_part, splitted], dim=0)
            return torch.nn.Parameter(new_p, requires_grad=p.requires_grad)

        elif name == "opacities":
            if revised_opacity:
                # revised => 1 - sqrt(1 - sigmoid(...))
                sigm = torch.sigmoid(mother_part)
                new_sigm = 1.0 - torch.sqrt(1.0 - sigm)
                splitted = torch.logit(new_sigm).repeat(2)
            else:
                # normal => replicate mother
                splitted = mother_part.repeat(2)
            new_p = torch.cat([father_part, splitted], dim=0)
            return torch.nn.Parameter(new_p, requires_grad=p.requires_grad)

        else:
            # default: replicate mother row 2x
            # shape: [old_count, ...]
            splitted = mother_part.repeat(2, *[1] * (p.ndim - 1))
            new_p = torch.cat([father_part, splitted], dim=0)
            return torch.nn.Parameter(new_p, requires_grad=p.requires_grad)

    ####################################################
    # 3) Build optimizer_fn for partial inheritance
    ####################################################
    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        """
        For each key in the optimizer state (e.g. 'exp_avg', 'exp_avg_sq', etc.),
        produce the new per-Gaussian array with partial inheritance.
        """
        if not isinstance(v, torch.Tensor):
            return v
        if v.dim() == 0 or v.shape[0] != old_count:
            return v

        father_vals = v[rest]
        mother_vals = v[sel]

        if key == "exp_avg":
            c1 = alpha_g * mother_vals
            c2 = alpha_g * mother_vals
        elif key == "exp_avg_sq":
            c1 = (alpha_g ** 2) * mother_vals
            c2 = (alpha_g ** 2) * mother_vals
        else:
            # e.g. zero them out or do alpha_t if you have 'lifespan'
            c1 = torch.zeros_like(mother_vals)
            c2 = torch.zeros_like(mother_vals)

        splitted = torch.cat([c1, c2], dim=0)
        return torch.cat([father_vals, splitted], dim=0)

    ####################################################
    # 4) Actually update the params + optimizer states
    ####################################################
    def _update_param_with_optimizer(
            param_fn: Callable[[str, torch.Tensor], torch.Tensor],
            optimizer_fn: Callable[[str, torch.Tensor], torch.Tensor],
            params: Dict[str, torch.nn.Parameter],
            optimizers: Dict[str, torch.optim.Optimizer],
            names: Union[List[str], None] = None,
    ):
        if names is None:
            names = list(params.keys())
        for name in names:
            old_p = params[name]
            new_p = param_fn(name, old_p)
            params[name] = new_p

            if name in optimizers:
                opt = optimizers[name]
                # Typically 1 param group
                for group in opt.param_groups:
                    if old_p in group["params"]:
                        group["params"].remove(old_p)
                    if new_p not in group["params"]:
                        group["params"].append(new_p)

                if old_p in opt.state:
                    old_state = opt.state.pop(old_p)
                    new_state = {}
                    for k_, v_ in old_state.items():
                        new_state[k_] = optimizer_fn(k_, v_)
                    opt.state[new_p] = new_state

    # Call the update
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names=None)

    ####################################################
    # 5) Update ephemeral states in `state`
    ####################################################
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.shape[0] != old_count:
            continue  # mismatch => skip

        father_part = v[rest]
        mother_part = v[sel]
        zero_part = torch.zeros_like(mother_part)
        splitted = torch.cat([zero_part, zero_part], dim=0)  # shape [2*N_sel, ...]
        state[k] = torch.cat([father_part, splitted], dim=0)

@torch.no_grad()
def split(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    revised_opacity: bool = False,
):
    """Inplace split the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to split the Gaussians.
        revised_opacity: Whether to use revised opacity formulation
          from arXiv:2404.06109. Default: False.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]

    scales = torch.exp(params["scales"][sel])
    quats = F.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales,
        torch.randn(2, len(scales), 3, device=device),
    )  # [2, N, 3]

    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
        elif name == "scales":
            p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
        elif name == "opacities" and revised_opacity:
            new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
            p_split = torch.logit(new_opacities).repeat(repeats)  # [2N]
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
        params: A dictionary of parameters (e.g., "means", "scales", "opacities", "quats").
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

    # Compute dominant axis
    scales_log = params["scales"][sel]  # [N, 3], log-scalings
    max_indices = torch.argmax(scales_log, dim=1)  # [N]
    scales = torch.exp(scales_log)  # [N, 3]
    offset_magnitudes = 1.5 * scales[torch.arange(len(sel)), max_indices]  # [N]

    # Deterministic offsets along dominant axis
    signs = torch.tensor([1.0, -1.0], device=device).view(2, 1)  # [2, 1]
    offsets_local = torch.zeros(2, len(sel), 3, device=device)
    offsets_local[torch.arange(2)[:, None], torch.arange(len(sel)), max_indices] = signs * offset_magnitudes

    # Rotate offsets to world space (assuming normalized_quat_to_rotmat is defined elsewhere)
    quats = F.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    offsets_world = torch.einsum("nij,bnj->bni", rotmats, offsets_local)  # [2, N, 3]
    new_means = (params["means"][sel].unsqueeze(0) + offsets_world).reshape(-1, 3)  # [2N, 3]

    # Adjust scaling
    log_0_5 = math.log(0.5)
    log_0_85 = math.log(0.85)
    adj = torch.full((len(sel), 3), log_0_85, device=device)
    adj[torch.arange(len(sel)), max_indices] = log_0_5
    new_scales = scales_log + adj  # [N, 3]
    new_scales = new_scales.repeat(2, 1)  # [2N, 3]

    def param_fn(name: str, p: Tensor) -> Tensor:
        """Update parameter tensors with split values."""
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            p_split = new_means  # [2N, 3]
        elif name == "scales":
            p_split = new_scales  # [2N, 3]
        elif name == "opacities":
            opacity = torch.sigmoid(p[sel])
            if revised_opacity:
                new_opacity = 1.0 - torch.sqrt(1.0 - opacity)
            else:
                new_opacity = 0.6 * opacity
            p_split_logit = torch.logit(new_opacity)
            if p.dim() == 1:
                p_split = p_split_logit.repeat(2)  # [2N]
            else:
                p_split = p_split_logit.repeat(2, 1)  # [2N, 1]
        else:
            p_split = p[sel].repeat(repeats)  # Adapts to p's dimensions
        p_new = torch.cat([p[rest], p_split], dim=0)
        p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        return p_new

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        """Update optimizer state."""
        v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
        return torch.cat([v[rest], v_split], dim=0)

    # Update parameters and optimizers (assuming _update_param_with_optimizer is defined)
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
    opacities = torch.sigmoid(params["opacities"].flatten())
    alpha_weight = (1.0 - opacities).pow(100)

    # 2) Compute local covariance for each Gaussian (orientation & scale)
    scales = torch.exp(params["scales"])
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

    # 5) In-place add to the means
    params["means"].add_(noise)

import math
import torch
from dataclasses import dataclass
from typing import Any, Dict, Union
from typing_extensions import Literal

from .base import Strategy
from .ops import remove, duplicate, split, split_edc
from .ops import inject_noise_to_position_new
from .ops import _update_param_with_optimizer

def multinomial_sample(weights: torch.Tensor, n: int) -> torch.Tensor:
    """
    Sample `n` indices from [0..len(weights)-1] with probabilities
    proportional to `weights`. Returns a LongTensor of shape [n].
    """
    weights = weights.clamp_min(1e-32)
    dist = torch.distributions.Categorical(probs=weights)
    return dist.sample((n,))

@dataclass
class DefaultStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper but with partial updates
    to mimic the Rust changes, e.g. alpha-based pruning, optional shrink+offset logic, etc.
    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.00008
    growth_stop_iter: int = 15_000

    min_opacity: float = 0.005
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    max_count: int = 10_000_000
    noise_lr: float = 5e4
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = True
    verbose: bool = False
    binoms: torch.Tensor = None
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        self.binoms = binoms
        return {"grad2d": None, "count": None, "scene_scale": scene_scale, "radii": None}

    def check_sanity(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
    ):
        super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
            self,
            params: Dict[str, torch.nn.Parameter],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
            info: Dict[str, Any],
    ):
        assert self.key_for_gradient in info, "Missing 2D means from info."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
            self,
            params: Dict[str, torch.nn.Parameter],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
            lr: float,
            info: Dict[str, Any],
            packed: bool = False,
    ):
        self.binoms = self.binoms.to(params["means"].device)
        self._update_state(params, state, info, packed=packed)

        # ----------------------------------------------------------------------------
        # CHANGED: Use (1 - alpha)^100 weighting for noise injection
        # ----------------------------------------------------------------------------
        if step < self.refine_stop_iter:
            inject_noise_to_position_new(
                params=params,
                optimizers=optimizers,
                state=state,
                scaler=lr * self.noise_lr,
            )
        # ----------------------------------------------------------------------------

        # Possibly refine
        if (step >= self.refine_start_iter) and (step < self.refine_stop_iter) and (step % self.refine_every == 0):
            n_prune = self._prune_gs(params, optimizers, state)
            n_added = 0
            if n_prune > 0:
                n_added_prune = self._replace_pruned(params, optimizers, state, n_prune)
                n_added += n_added_prune

            # Possibly grow from high grad
            n_grow = self._grow_gs(params, optimizers, state, step)
            n_added += n_grow

            if self.verbose:
                eff_growth = n_added - n_prune
                print(
                    f"[Refine@step={step}] pruned={n_prune} added={n_added} "
                    f"-> effective_growth={eff_growth} #splats={len(params['means'])}"
                )

            # reset stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if state["radii"] is not None:
                state["radii"].zero_()

    def _update_state(
            self,
            params: Dict[str, torch.nn.Parameter],
            state: Dict[str, Any],
            info: Dict[str, Any],
            packed: bool = False,
    ):
        for key in ["width", "height", "n_cameras", "radii", "gaussian_ids", self.key_for_gradient]:
            assert key in info, f"{key} is required but missing."

        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()

        w, h, ncams = info["width"], info["height"], info["n_cameras"]
        grads[..., 0] *= (w / 2.0 * ncams)
        grads[..., 1] *= (h / 2.0 * ncams)

        n_gaussian = len(params["means"])
        if state["grad2d"] is None or state["grad2d"].shape[0] != n_gaussian:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None or state["count"].shape[0] != n_gaussian:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if state["radii"] is None or state["radii"].shape[0] != n_gaussian:
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        if packed:
            gs_ids = info["gaussian_ids"]
            radii = info["radii"]
        else:
            sel = info["radii"] > 0.0
            gs_ids = torch.where(sel)[1]
            grads = grads[sel]
            radii = info["radii"][sel]

        norms = grads.norm(dim=-1)
        state["grad2d"].index_add_(0, gs_ids, norms)
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))

        # keep the max radii for each splat
        max_wh = float(max(w, h))
        new_radii = radii / max_wh
        state["radii"][gs_ids] = torch.maximum(state["radii"][gs_ids], new_radii)

    @torch.no_grad()
    def _prune_gs(
            self,
            params: Dict[str, torch.nn.Parameter],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
    ) -> int:
        alpha = torch.sigmoid(params["opacities"])
        is_prune = alpha < self.prune_opa
        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
        return n_prune

    def importance_prune(
            self,
            params: Dict[str, torch.nn.Parameter],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            mask,
    ):
        remove(params=params, optimizers=optimizers, state=state, mask=mask)

    @torch.no_grad()
    def _replace_pruned(
            self,
            params: Dict[str, torch.nn.Parameter],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, torch.Tensor],
            n_prune: int,
    ) -> int:
        """
        Replace pruned splats by sampling from alpha distribution of the remaining,
        then performing a shrink-and-offset logic for certain parameters:
            - means: shift in-place, then duplicate with mirrored offset
            - scales: subtract ln(sqrt(2)) in-place, then duplicate
            - opacities: alpha_new = 1 - sqrt(1 - alpha), then duplicate
            - quats: just copy old -> new

        For ANY other param name, we mimic 'sample_add': first do
            p[sampled_idxs] = something
        then cat p[sampled_idxs].
        Finally, we expand the 'state' dictionary by concatenating zeros for each newly added Gaussian.
        """
        old_count = len(params["means"])
        if old_count == 0 or n_prune <= 0:
            return 0

        alpha = torch.sigmoid(params["opacities"])
        weights = alpha.clone().clamp_min(1e-32)
        device = params["means"].device

        # 1) Sample `n_prune` from alpha distribution
        chosen_inds = multinomial_sample(weights, n_prune)  # shape [n_prune]
        if chosen_inds.numel() == 0:
            return 0

        # Basic shrink/offset parameters
        scale_offset = math.log(math.sqrt(2.0))  # ~0.3466
        noise_std = 0.5

        # 2) param_fn: Modify p[sampled_inds] in-place, then cat p[sampled_inds] to the end
        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            p_old = p.clone()
            N = p_old.shape[0]
            c = chosen_inds.shape[0]

            if c == 0:
                return torch.nn.Parameter(p_old, requires_grad=p.requires_grad)

            if name == "means":
                # shape [N, 3]
                old_means = p_old[chosen_inds]
                s_log = params["scales"].data[chosen_inds]
                s_linear = torch.exp(s_log)
                rand_delta = torch.randn_like(old_means, device=device) * noise_std * s_linear
                # In-place update: subtract delta
                p_old[chosen_inds] = old_means - rand_delta
                # Append new row = old_means + delta
                appended = old_means + rand_delta
                p_new = torch.cat([p_old, appended], dim=0)

            elif name == "scales":
                # shape [N, 3] in log space
                old_log_scales = p_old[chosen_inds]
                # Subtract ln(sqrt(2)) in-place
                new_log = old_log_scales - scale_offset
                p_old[chosen_inds] = new_log
                # Duplicate
                p_new = torch.cat([p_old, new_log], dim=0)

            elif name == "opacities":
                o_raw_old = p_old[chosen_inds]
                alpha_old = torch.sigmoid(o_raw_old)
                alpha_new = 1.0 - torch.sqrt(1.0 - alpha_old.clamp(max=0.9999999))
                raw_new = (alpha_new / (1.0 - alpha_new + 1e-24)).log()
                p_old[chosen_inds] = raw_new
                p_new = torch.cat([p_old, raw_new], dim=0)

            elif name == "quats":
                old_quats = p_old[chosen_inds]
                p_new = torch.cat([p_old, old_quats], dim=0)

            else:
                chosen_chunk = p_old[chosen_inds]
                p_new = torch.cat([p_old, chosen_chunk], dim=0)

            return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

        # 3) optimizer_fn: Append zeros for newly added rows in optimizer state
        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            if v.shape[0] == 0:
                return v
            new_part = torch.zeros_like(v[chosen_inds])
            v_new = torch.cat([v, new_part], dim=0)
            return v_new

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

        # 5) Expand the user state
        new_count = len(params["means"]) - old_count
        if new_count > 0:
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == old_count:
                    extra_shape = (len(chosen_inds), *v.shape[1:])
                    v_new = torch.zeros(extra_shape, dtype=v.dtype, device=v.device)
                    state[k] = torch.cat([v, v_new], dim=0)

        return new_count

    @torch.no_grad()
    def _grow_gs(
            self,
            params: Dict[str, torch.nn.Parameter],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
    ) -> int:
        if step >= self.growth_stop_iter or len(params["means"]) >= self.max_count:
            return 0

        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (torch.exp(params["scales"]).max(dim=-1).values <= self.grow_scale3d * state["scene_scale"])
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter and state["radii"] is not None:
            is_split |= (state["radii"] > self.grow_scale2d)
        n_split = is_split.sum().item()

        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # after duplication, the new rows are appended at the end, so we have to
        # extend is_split by that many “False” entries
        is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)], dim=0)
        if n_split > 0:
            split_edc(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        return (n_dupli + n_split)

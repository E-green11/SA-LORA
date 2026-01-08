from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from peft.tuners import lora
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (EvalPrediction, PreTrainedModel,
                                  PreTrainedTokenizerBase, TrainerCallback)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled, logging

# --- DyLoRA START ---
# DyLoRA: additional imports
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
# --- DyLoRA END ---

from dataclasses import dataclass, field
from functools import reduce
# from typing import Callable, Dict, List, Optional, Tuple, Union # original imports replaced above

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from peft.tuners import lora
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (EvalPrediction, PreTrainedModel,
                                  PreTrainedTokenizerBase, TrainerCallback)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled, logging

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

logger = logging.get_logger(__name__)


@dataclass
class LoraPlusTrainingArguments(TrainingArguments):
    loraplus_lr_ratio: Optional[float] = field(
        default=None, metadata={"help": "loraplus learning rate ratio lr_B / lr_A."}
    )
    loraplus_lr_embedding: Optional[float] = field(
        default=1e-6,
        metadata={"help": "loraplus learning rate for lora embedding layers."},
    )
    # --- SA-LoRA START ---
    use_sa_lora: bool = field(
        default=False,
        metadata={"help": "Enable Structure-Aware LoRA per-layer LR multipliers."},
    )
    sa_metric: str = field(
        default="stable_rank",
        metadata={
            "help": "Spectral metric to compute per-layer scores: stable_rank | spectral_entropy | cond",
        },
    )
    sa_min_mult: float = field(
        default=0.5,
        metadata={"help": "Minimum LR multiplier for any LoRA layer when SA-LoRA is enabled."},
    )
    sa_max_mult: float = field(
        default=2.0,
        metadata={"help": "Maximum LR multiplier for any LoRA layer when SA-LoRA is enabled."},
    )
    sa_power: float = field(
        default=1.0,
        metadata={"help": "Nonlinear mapping power. 1.0 is linear; >1 emphasizes higher-score layers."},
    )
    sa_apply_to: str = field(
        default="both",
        metadata={"help": "Apply multipliers to A, B, or both LoRA params."},
    )
    sa_normalize_mean: bool = field(
        default=True,
        metadata={"help": "Rescale per-layer multipliers so their mean is 1.0."},
    )
    sa_warmup_steps: int = field(
        default=1000,
        metadata={"help": "Warm up SA multipliers over first N optimizer steps; 0 to disable."},
    )
    # Grad-based calibration (early gradient energy) can be blended with spectral metric
    sa_grad_calibrate_steps: int = field(
        default=0,
        metadata={"help": "Accumulate per-layer gradient energy for first N steps to calibrate multipliers. 0 disables."},
    )
    sa_grad_power: float = field(
        default=1.0,
        metadata={"help": "Exponent for mapping grad energy to multiplier before normalization."},
    )
    sa_grad_blend: float = field(
        default=0.0,
        metadata={"help": "Blend weight in [0,1] between spectral (1-w) and grad-based (w) multipliers."},
    )
    # Online EMA calibration and depth prior
    sa_online_ema: bool = field(
        default=False,
        metadata={"help": "Enable online EMA calibration of per-layer multipliers during training."},
    )
    sa_ema_beta: float = field(
        default=0.9,
        metadata={"help": "EMA smoothing factor for online gradient energy per layer."},
    )
    sa_ema_update_every: int = field(
        default=400,
        metadata={"help": "Update online EMA and refresh multipliers every N optimizer steps."},
    )
    sa_depth_prior: bool = field(
        default=False,
        metadata={"help": "Blend a depth-aware prior into multipliers (higher layers slightly larger)."},
    )
    sa_depth_prior_weight: float = field(
        default=0.1,
        metadata={"help": "Weight in [0,1] for depth prior when blending with spectral/grad multipliers."},
    )
    # --- SA-LoRA END ---
    
    # --- DyLoRA START ---
    # Add DyLoRA-related training arguments
    use_dylora: bool = field(
        default=False,
        metadata={"help": "Whether to use Dynamic LoRA+ (DyLoRA). Overrides loraplus_lr_ratio if True."}
    )
    dylora_beta: float = field(
        default=0.99,
        metadata={"help": "Exponential Moving Average (EMA) smoothing factor for DyLoRA ratio (beta)."}
    )
    dylora_min_lr_ratio: float = field(
        default=1.0,
        metadata={"help": "Minimum value (floor) for the dynamic LoRA ratio."}
    )
    dylora_max_lr_ratio: float = field(
        default=100.0,
        metadata={"help": "Maximum value (ceiling) for the dynamic LoRA ratio."}
    )
    dylora_update_every: int = field(
        default=1,
        metadata={"help": "Update the dynamic ratio every N optimization steps."}
    )
    # --- DyLoRA END ---


def get_module(name, opt_model):
    """
    Retrieve a module from a model using its parameter name.
    Args:
        name (str): Full name of the parameter, typically including module path.
        opt_model (torch.nn.Module): The model from which to retrieve the module.

    Returns:
        Module corresponding to the given name.
    """
    parent_idx = 2 if "lora" in name else 1
    module_names = name.split(sep=".")[:-parent_idx]
    module = reduce(getattr, module_names, opt_model)
    return module


def create_loraplus_optimizer(
    opt_model,
    optimizer_cls,
    optimizer_kwargs,
    loraplus_lr_ratio,
    loraplus_lr_embedding=None,
    # --- DyLoRA START ---
    # Add an optional parameter to accept the trainer instance
    trainer_instance: Optional[Trainer] = None,
    # --- DyLoRA END ---
):
    """
    Creates an optimizer for the given model, applying LoRA-specific learning rate adjustments to different parameter groups.

    Args:
        opt_model (torch.nn.Module): The model for which the optimizer is being created.
        optimizer_cls (class): The class of the optimizer to be used (e.g., torch.optim.Adam).
        optimizer_kwargs (dict): A dictionary of keyword arguments for the optimizer's initialization.
        loraplus_lr_ratio (float): The learning rate ratio to be applied to LoRA parameters.
        loraplus_lr_embedding (float, optional): A specific learning rate for embedding parameters, with a default value if not provided.
    # --- DyLoRA START ---
        trainer_instance (Trainer, optional): If using DyLoRA, pass the trainer instance to store param groups.
    # --- DyLoRA END ---

    Returns:
        An instance of the specified optimizer class configured with the model's parameters organized into groups with custom learning rates.
    """

    # --- DyLoRA START ---
    # When using DyLoRA, loraplus_lr_ratio may be dynamic; only assert when not using DyLoRA
    if not (trainer_instance and trainer_instance.args.use_dylora):
        assert loraplus_lr_ratio is not None, "loraplus_lr_ratio must be provided when not using DyLoRA."
    # --- DyLoRA END ---

    if loraplus_lr_embedding is None:
        loraplus_lr_embedding = 1e-6

    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }

    # --- SA-LoRA START ---
    # Compute spectral score per LoRA target module and generate per-module LR multipliers
    def _module_key_from_param_name(param_name: str) -> str:
        parts = param_name.split(".")
        if "lora_A" in parts:
            idx = parts.index("lora_A")
            return ".".join(parts[:idx])
        if "lora_B" in parts:
            idx = parts.index("lora_B")
            return ".".join(parts[:idx])
        # fallback: drop last token (param name like weight, bias)
        return ".".join(parts[:-1])

    def _compute_spectral_score(weight: torch.Tensor, metric: str) -> float:
        with torch.no_grad():
            w = weight.detach().float().cpu()
            # Only handle matrices. For vectors, return minimal score.
            if w.ndim < 2:
                return 1.0
            try:
                # singular values descending
                s = torch.linalg.svdvals(w)
            except Exception:
                # fall back to power iteration for top-1; still need Fro norm
                fro_sq = (w * w).sum().item()
                # rough spectral norm by one power iteration
                v = torch.randn(w.shape[1], device=w.device)
                for _ in range(10):
                    v = torch.mv(w.t(), torch.mv(w, v))
                    v = v / (v.norm() + 1e-12)
                spec = torch.sqrt(torch.mv(w, v).pow(2).sum()).item() + 1e-12
                return float(fro_sq / (spec * spec))

            if metric == "stable_rank":
                fro_sq = torch.sum(s * s).item()
                spec_sq = (s.max().item() ** 2) + 1e-12
                return float(fro_sq / spec_sq)
            elif metric == "spectral_entropy":
                s = s.clamp_min(1e-12)
                p = s / s.sum()
                ent = -(p * (p.log())).sum().item()
                # normalize by log(n)
                ent_norm = ent / (float(torch.log(torch.tensor(float(len(s)))).item()) + 1e-12)
                return float(ent_norm)
            elif metric == "cond":
                s = s.clamp_min(1e-12)
                return float((s.max() / s.min()).item())
            else:
                # default to stable_rank
                fro_sq = torch.sum(s * s).item()
                spec_sq = (s.max().item() ** 2) + 1e-12
                return float(fro_sq / spec_sq)

    sa_enabled = bool(trainer_instance and trainer_instance.args.use_sa_lora)
    sa_metric = None
    per_module_multiplier: Dict[int, float] = {}
    per_module_name: Dict[int, str] = {}
    if sa_enabled:
        sa_metric = trainer_instance.args.sa_metric
        raw_scores: List[Tuple[int, float]] = []  # (module_id, score)

    # Buckets to collect parameters per module for creating groups later
    sa_buckets: Dict[int, Dict[str, Dict[str, torch.nn.Parameter]]] = {}

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue

        module = get_module(name, opt_model)
        if isinstance(module, lora.Embedding):
            param_groups["embedding"][name] = param
            continue

        is_lora_param = ("lora_A" in name) or ("lora_B" in name)

        if sa_enabled and is_lora_param:
            key = _module_key_from_param_name(name)
            mod_id = id(module)
            per_module_name[mod_id] = key
            if mod_id not in sa_buckets:
                sa_buckets[mod_id] = {"A": {}, "B": {}, "B_no_decay": {}}
                # Compute this module's spectral score (once)
                try:
                    score = _compute_spectral_score(module.weight, trainer_instance.args.sa_metric)
                except Exception:
                    score = 1.0
                if sa_enabled:
                    raw_scores.append((mod_id, float(score)))

            if "lora_B" in name or param.ndim == 1:
                if name in decay_parameters:
                    sa_buckets[mod_id]["B"][name] = param
                else:
                    sa_buckets[mod_id]["B_no_decay"][name] = param
            else:
                sa_buckets[mod_id]["A"][name] = param
        else:
            if "lora_B" in name or param.ndim == 1:
                if name in decay_parameters:
                    param_groups["groupB"][name] = param
                else:
                    param_groups["groupB_no_decay"][name] = param
            else:
                param_groups["groupA"][name] = param
            
    # --- DyLoRA START ---
    # If DyLoRA is enabled, store parameter lists on the trainer instance
    if trainer_instance is not None and trainer_instance.args.use_dylora:
        logger.info("DyLoRA is enabled. Storing parameter groups.")
        # store lists of parameters, not dicts
        trainer_instance.dylora_groupA_params = list(param_groups["groupA"].values())
        trainer_instance.dylora_groupB_params = list(param_groups["groupB"].values()) + list(param_groups["groupB_no_decay"].values())
        
        # Record indices of B groups in the optimizer to update later (populated after creating param groups)
        trainer_instance.dylora_groupB_indices = []
        trainer_instance.dylora_B_to_A_index = {}

        # Use dylora_current_lr_ratio as the initial ratio
        loraplus_lr_ratio = trainer_instance.dylora_current_lr_ratio
    # --- DyLoRA END ---

    assigned_param_groups = ""
    for group in param_groups:
        assigned_param_groups += f"{group}\n {list(param_groups[group].keys())}\n\n"
    logger.info(assigned_param_groups)

    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    optimizer_grouped_parameters = []

    # 1) Global (non-SA) group A
    optimizer_grouped_parameters.append(
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        }
    )

    # 2) embedding group
    optimizer_grouped_parameters.append(
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": weight_decay,
            "lr": loraplus_lr_embedding,
        }
    )

    # 3) SA-LoRA: create per-module A/B groups and apply multipliers
    sa_a_indices: Dict[int, int] = {}
    if sa_enabled and len(sa_buckets) > 0:
        # Generate multipliers (linearly scaled to [min,max])
        scores = [s for (_, s) in raw_scores]
        s_min = min(scores) if len(scores) > 0 else 1.0
        s_max = max(scores) if len(scores) > 0 else 1.0
        span = (s_max - s_min) + 1e-12
        min_m = trainer_instance.args.sa_min_mult
        max_m = trainer_instance.args.sa_max_mult
        power = trainer_instance.args.sa_power

        for mod_id, score in raw_scores:
            norm = max(0.0, min(1.0, (score - s_min) / span))
            mult = min_m + (max_m - min_m) * (norm ** power)
            per_module_multiplier[mod_id] = float(mult)

        # Depth prior: slight gain based on encoder.layer.X numbering (higher layers larger)
        if trainer_instance.args.sa_depth_prior:
            # Collect and sort layer ids
            layer_ids: List[int] = []
            mod_id_to_layer: Dict[int, int] = {}
            for mod_id, name in per_module_name.items():
                lid = -1
                parts = name.split(".")
                if "layer" in parts:
                    try:
                        idx = parts.index("layer") + 1
                        lid = int(parts[idx])
                    except Exception:
                        lid = -1
                mod_id_to_layer[mod_id] = lid
                if lid >= 0:
                    layer_ids.append(lid)
            if len(layer_ids) > 0:
                Lmin, Lmax = min(layer_ids), max(layer_ids)
                Lspan = float((Lmax - Lmin) or 1)
                w = max(0.0, min(1.0, trainer_instance.args.sa_depth_prior_weight))
                for mod_id in per_module_multiplier.keys():
                    lid = mod_id_to_layer.get(mod_id, -1)
                    if lid >= 0:
                        depth_norm = (lid - Lmin) / Lspan
                        prior = 1.0 + 0.2 * depth_norm  # up to +20% for highest layers
                        per_module_multiplier[mod_id] = (
                            (1.0 - w) * per_module_multiplier[mod_id] + w * prior
                        )
                        per_module_multiplier[mod_id] = float(max(min_m, min(max_m, per_module_multiplier[mod_id])))

        # If grad calibration enabled, initialize containers to accumulate and fuse later in training_step
        if trainer_instance is not None and trainer_instance.args.sa_grad_calibrate_steps > 0:
            if not hasattr(trainer_instance, 'sa_grad_energy_A'):
                trainer_instance.sa_grad_energy_A = {id_: 0.0 for id_ in sa_buckets.keys()}
                trainer_instance.sa_grad_energy_B = {id_: 0.0 for id_ in sa_buckets.keys()}
                trainer_instance.sa_grad_steps = 0
                trainer_instance.sa_module_id_to_aidx = {}

        # Initialize online EMA
        if trainer_instance is not None and trainer_instance.args.sa_online_ema:
            if not hasattr(trainer_instance, 'sa_online_ema_energy'):
                trainer_instance.sa_online_ema_energy = {id_: 0.0 for id_ in sa_buckets.keys()}

        # Mean-one normalization (preserve overall LR mean)
        if trainer_instance.args.sa_normalize_mean and len(per_module_multiplier) > 0:
            mean_mult = sum(per_module_multiplier.values()) / len(per_module_multiplier)
            if mean_mult > 0:
                for k in list(per_module_multiplier.keys()):
                    per_module_multiplier[k] = float(per_module_multiplier[k] / mean_mult)
                    # Clip to [min,max] after normalization
                    per_module_multiplier[k] = float(max(min_m, min(max_m, per_module_multiplier[k])))

        # Add parameter groups per module
        for mod_id, buckets in sa_buckets.items():
            mult = per_module_multiplier.get(mod_id, 1.0)

            apply_to = trainer_instance.args.sa_apply_to.lower()
            apply_A = apply_to in ("a", "both")
            apply_B = apply_to in ("b", "both")

            # A group
            a_lr = lr * (mult if apply_A else 1.0)
            a_idx = len(optimizer_grouped_parameters)
            optimizer_grouped_parameters.append(
                {"params": list(buckets["A"].values()), "weight_decay": weight_decay, "lr": a_lr}
            )
            sa_a_indices[mod_id] = a_idx
            if trainer_instance is not None and trainer_instance.args.sa_grad_calibrate_steps > 0:
                trainer_instance.sa_module_id_to_aidx[mod_id] = a_idx

            # B (with weight decay)
            b_lr = (lr * loraplus_lr_ratio) * (mult if apply_B else 1.0)
            b_idx = len(optimizer_grouped_parameters)
            optimizer_grouped_parameters.append(
                {"params": list(buckets["B"].values()), "weight_decay": weight_decay, "lr": b_lr}
            )
            # B (no weight decay)
            b2_idx = len(optimizer_grouped_parameters)
            optimizer_grouped_parameters.append(
                {"params": list(buckets["B_no_decay"].values()), "weight_decay": 0.0, "lr": b_lr}
            )

            # Record mapping from B to A group indices (for DyLoRA and SA warmup)
            if trainer_instance is not None:
                if not hasattr(trainer_instance, "dylora_groupB_indices"):
                    trainer_instance.dylora_groupB_indices = []
                if not hasattr(trainer_instance, "dylora_B_to_A_index"):
                    trainer_instance.dylora_B_to_A_index = {}
                if len(buckets["B"]) > 0:
                    trainer_instance.dylora_groupB_indices.append(b_idx)
                    trainer_instance.dylora_B_to_A_index[b_idx] = a_idx
                if len(buckets["B_no_decay"]) > 0:
                    trainer_instance.dylora_groupB_indices.append(b2_idx)
                    trainer_instance.dylora_B_to_A_index[b2_idx] = a_idx

    # 4) Non-SA B groups (if any remain)
    if len(param_groups["groupB"]) > 0:
        b_idx = len(optimizer_grouped_parameters)
        optimizer_grouped_parameters.append(
            {
                "params": list(param_groups["groupB"].values()),
                "weight_decay": weight_decay,
                "lr": lr * loraplus_lr_ratio,
            }
        )
        if trainer_instance is not None:
            trainer_instance.dylora_groupB_indices.append(b_idx)
            # Use global A group (index 0) as reference
            trainer_instance.dylora_B_to_A_index[b_idx] = 0
    if len(param_groups["groupB_no_decay"]) > 0:
        b2_idx = len(optimizer_grouped_parameters)
        optimizer_grouped_parameters.append(
            {
                "params": list(param_groups["groupB_no_decay"].values()),
                "weight_decay": 0.0,
                "lr": lr * loraplus_lr_ratio,
            }
        )
        if trainer_instance is not None:
            trainer_instance.dylora_groupB_indices.append(b2_idx)
            trainer_instance.dylora_B_to_A_index[b2_idx] = 0

    # Log SA multipliers for debugging
    if sa_enabled and len(per_module_multiplier) > 0:
        debug_lines = [
            f"{per_module_name[mid]} -> mult={per_module_multiplier[mid]:.4f}"
            for mid in per_module_multiplier
        ]
        logger.info("[SA-LoRA] Layer LR multipliers:\n" + "\n".join(debug_lines))
        # Log distribution stats of raw spectral scores to aid tuning
        if 'raw_scores' in locals() and len(raw_scores) > 0:
            raw_vals = [float(s) for (_, s) in raw_scores]
            s_min, s_max = min(raw_vals), max(raw_vals)
            s_mean = sum(raw_vals) / len(raw_vals)
            logger.info(f"[SA-LoRA] spectral scores stats -> min={s_min:.4f}, mean={s_mean:.4f}, max={s_max:.4f}")

    # Save SA info to trainer to support warmup / syncing B groups
    if sa_enabled and trainer_instance is not None:
        trainer_instance.sa_enabled = True
        trainer_instance.sa_warmup_steps = max(0, int(trainer_instance.args.sa_warmup_steps))
        trainer_instance.sa_a_group_indices = []
        trainer_instance.sa_a_multiplier = {}
        for mod_id, a_idx in ({} if not (sa_enabled and len(sa_buckets) > 0) else {k: v for k, v in sa_a_indices.items()}).items():
            trainer_instance.sa_a_group_indices.append(a_idx)
            trainer_instance.sa_a_multiplier[a_idx] = per_module_multiplier.get(mod_id, 1.0)

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum(
                    {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                )
                logger.info(f"skipped {module}: {skipped/2**20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped/2**20}M params")

    return optimizer


class LoraPlusTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: LoraPlusTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        assert isinstance(
            args, LoraPlusTrainingArguments
        ), "args must be of type LoraPlusTrainingArguments"
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        
        # --- DyLoRA START ---
        # Initialize DyLoRA state
        if self.args.use_dylora:
            # Start from dylora_min_lr_ratio instead of 1.0
            self.dylora_current_lr_ratio = self.args.dylora_min_lr_ratio
            self.dylora_groupA_params = []
            self.dylora_groupB_params = []
            self.dylora_groupB_indices = []
            logger.info(f"DyLoRA enabled. Initial ratio set to {self.dylora_current_lr_ratio}")
        # --- DyLoRA END ---

    def create_optimizer(self):
        """
        Overrides the method to create an optimizer with LoRA+ specific adjustments.
        """
        # --- DyLoRA START ---
        # Modify condition to include use_dylora
        if self.args.loraplus_lr_ratio is None and not self.args.use_dylora:
            return super().create_optimizer()
        # --- DyLoRA END ---

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            # --- DyLoRA START ---
            # If using DyLoRA, loraplus_lr_ratio is only the *initial* ratio; otherwise it's static
            if self.args.use_dylora:
                loraplus_lr_ratio = self.dylora_current_lr_ratio
                logger.info(f"DyLoRA: create_optimizer using initial ratio: {loraplus_lr_ratio}")
            else:
                loraplus_lr_ratio = getattr(self.args, "loraplus_lr_ratio", None)
            # --- DyLoRA END ---
            
            loraplus_lr_embedding = getattr(self.args, "loraplus_lr_embedding", None)
            
            self.optimizer = create_loraplus_optimizer(
                opt_model,
                optimizer_cls,
                optimizer_kwargs,
                loraplus_lr_ratio,
                loraplus_lr_embedding,
                # --- DyLoRA START ---
                # Pass trainer instance to store parameter groups
                trainer_instance=self,
                # --- DyLoRA END ---
            )

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    # --- DyLoRA START ---
    # Override training_step to inject dynamic ratio computation.
    # This implementation is compatible with the rest of the codebase.
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # Use Accelerate backward (if available) to avoid conflicts with its GradScaler
        if getattr(self, "accelerator", None) is not None:
            self.accelerator.backward(loss)
        elif self.args.deepspeed:
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        # --- DyLoRA injection logic ---
        # Execute after backward() and before step().
        # Note: self.state.global_step increments after step(), so use (self.state.global_step + 1)
        if self.args.use_dylora and (self.state.global_step + 1) % self.args.dylora_update_every == 0:
            with torch.no_grad():
                # 1. Compute gradient norms for A and B groups
                # Create tensors on loss.device
                norm_A = torch.tensor(0.0, device=loss.device)
                for p in self.dylora_groupA_params:
                    if p.grad is not None:
                        # Use pow(2).sum() instead of norm()**2 for efficiency
                        norm_A += p.grad.detach().pow(2).sum()
                norm_A = torch.sqrt(norm_A)

                norm_B = torch.tensor(0.0, device=loss.device)
                for p in self.dylora_groupB_params:
                    if p.grad is not None:
                        norm_B += p.grad.detach().pow(2).sum()
                norm_B = torch.sqrt(norm_B)

            # 2. Compute updated ratio
            if norm_B > 1e-8:  # avoid division by zero
                current_ratio = (norm_A / norm_B).item()
                
                # 3. Apply EMA smoothing
                self.dylora_current_lr_ratio = (
                    self.args.dylora_beta * self.dylora_current_lr_ratio
                    + (1 - self.args.dylora_beta) * current_ratio
                )
                
                # 4. Clip ratio
                self.dylora_current_lr_ratio = max(
                    self.args.dylora_min_lr_ratio,
                    min(self.dylora_current_lr_ratio, self.args.dylora_max_lr_ratio)
                )

            # 5. Update B-group learning rates in the optimizer. When using SA-LoRA, each B group maps to its A group index
            for idx in self.dylora_groupB_indices:
                a_idx = getattr(self, 'dylora_B_to_A_index', {}).get(idx, 0)
                base_lr = self.optimizer.param_groups[a_idx]['lr']
                new_lr_B = base_lr * self.dylora_current_lr_ratio
                self.optimizer.param_groups[idx]['lr'] = new_lr_B
            
            # 7. Log DyLoRA metrics
            if self.args.logging_steps > 0 and (self.state.global_step + 1) % self.args.logging_steps == 0:
                    self.log({
                        "train/dylora_lr_ratio": self.dylora_current_lr_ratio,
                        "train/dylora_lr_B": new_lr_B if 'new_lr_B' in locals() else self.optimizer.param_groups[self.dylora_groupB_indices[0]]['lr'],
                        "train/dylora_norm_A": norm_A.item(),
                        "train/dylora_norm_B": norm_B.item()
                    })
        # --- DyLoRA injection end ---

        # SA-LoRA: gradient calibration and warmup
        # 1) Grad-based calibration: accumulate per-module A/B gradient energy for first N steps
        if getattr(self.args, 'sa_grad_calibrate_steps', 0) > 0 and (self.state.global_step + 1) <= self.args.sa_grad_calibrate_steps:
            with torch.no_grad():
                for a_idx in getattr(self, 'sa_a_group_indices', []):
                    energy = 0.0
                    for p in self.optimizer.param_groups[a_idx]['params']:
                        if p.grad is not None:
                            energy += float(p.grad.detach().pow(2).sum().item())
                    # Reverse-lookup module id via sa_module_id_to_aidx mapping
                if hasattr(self, 'sa_module_id_to_aidx'):
                    for mod_id, a_idx in self.sa_module_id_to_aidx.items():
                        # The above energy was accumulated per group; recompute per-module here for robustness
                        e_val = 0.0
                        for p in self.optimizer.param_groups[a_idx]['params']:
                            if p.grad is not None:
                                e_val += float(p.grad.detach().pow(2).sum().item())
                        self.sa_grad_energy_A[mod_id] = self.sa_grad_energy_A.get(mod_id, 0.0) + e_val
                # B-group energy is handled similarly (used for stability; A is currently used)
            self.sa_grad_steps = getattr(self, 'sa_grad_steps', 0) + 1

        # Online EMA: periodically update per-layer gradient energy and lightly refresh multipliers (mild, bounded, mean≈1)
        if getattr(self.args, 'sa_online_ema', False) and (self.state.global_step + 1) % max(1, int(self.args.sa_ema_update_every)) == 0:
            if hasattr(self, 'sa_module_id_to_aidx') and hasattr(self, 'sa_online_ema_energy'):
                beta = float(self.args.sa_ema_beta)
                with torch.no_grad():
                    # Update EMA energy
                    for mod_id, a_idx in self.sa_module_id_to_aidx.items():
                        e_val = 0.0
                        for p in self.optimizer.param_groups[a_idx]['params']:
                            if p.grad is not None:
                                e_val += float(p.grad.detach().pow(2).sum().item())
                        prev = float(self.sa_online_ema_energy.get(mod_id, 0.0))
                        self.sa_online_ema_energy[mod_id] = beta * prev + (1.0 - beta) * e_val

                        # Map EMA energy to small multipliers and fuse into existing target multipliers
                    ema_vals = list(self.sa_online_ema_energy.values())
                    if len(ema_vals) > 0:
                        g_min, g_max = min(ema_vals), max(ema_vals)
                        span = (g_max - g_min) + 1e-12
                        min_m = self.args.sa_min_mult
                        max_m = self.args.sa_max_mult
                        # Use a mild exponent and small blend weight to avoid large perturbations
                        pwr = 1.0
                        online_w = 0.2
                        base = self.optimizer.param_groups[0]['lr']
                        updated: Dict[int, float] = {}
                        for mod_id, a_idx in self.sa_module_id_to_aidx.items():
                            g = self.sa_online_ema_energy[mod_id]
                            norm = max(0.0, min(1.0, (g - g_min) / span))
                            ema_mult = min_m + (max_m - min_m) * (norm ** pwr)
                            spectral_mult = getattr(self, 'sa_a_multiplier', {}).get(a_idx, 1.0)
                            fused = (1.0 - online_w) * spectral_mult + online_w * ema_mult
                            updated[a_idx] = float(max(min_m, min(max_m, fused)))

                        # Mean normalization
                        if getattr(self.args, 'sa_normalize_mean', True):
                            mean_val = sum(updated.values()) / float(len(updated))
                            if mean_val > 0:
                                for a_idx in list(updated.keys()):
                                    updated[a_idx] = float(updated[a_idx] / mean_val)
                                    updated[a_idx] = float(max(min_m, min(max_m, updated[a_idx])))

                        # Write back target multipliers (warmup will interpolate towards targets) and lightly refresh current LRs
                        for a_idx, mult in updated.items():
                            if hasattr(self, 'sa_a_multiplier'):
                                self.sa_a_multiplier[a_idx] = mult
                            self.optimizer.param_groups[a_idx]['lr'] = base * mult
                        for b_idx, mapped_a in getattr(self, 'dylora_B_to_A_index', {}).items():
                            base_lr = self.optimizer.param_groups[mapped_a]['lr']
                            ratio = self.dylora_current_lr_ratio if getattr(self.args, 'use_dylora', False) else (getattr(self.args, 'loraplus_lr_ratio', 1.0) or 1.0)
                            self.optimizer.param_groups[b_idx]['lr'] = base_lr * ratio

                        vals = list(updated.values())
                        logger.info(f"[SA-LoRA] Online EMA update at step {self.state.global_step + 1}: min={min(vals):.4f}, mean={sum(vals)/len(vals):.4f}, max={max(vals):.4f}")

        # 2) After calibration completes, map gradient energy to multipliers and fuse into current SA multipliers (by adjusting A/B group LRs)
        if getattr(self.args, 'sa_grad_calibrate_steps', 0) > 0 and (self.state.global_step + 1) == self.args.sa_grad_calibrate_steps:
            if getattr(self, 'sa_grad_steps', 0) > 0 and hasattr(self, 'sa_module_id_to_aidx'):
                # Normalize gradient energy and map to [min,max]
                grad_vals = [v for v in self.sa_grad_energy_A.values()]
                if len(grad_vals) > 0:
                    g_min, g_max = min(grad_vals), max(grad_vals)
                    span = (g_max - g_min) + 1e-12
                    min_m = self.args.sa_min_mult
                    max_m = self.args.sa_max_mult
                    pwr = self.args.sa_grad_power
                    blend = max(0.0, min(1.0, self.args.sa_grad_blend))

                    base = self.optimizer.param_groups[0]['lr']
                    fused_by_aidx: Dict[int, float] = {}

                    # First compute fused target multipliers per module
                    for mod_id, a_idx in self.sa_module_id_to_aidx.items():
                        g = self.sa_grad_energy_A.get(mod_id, 0.0)
                        norm = max(0.0, min(1.0, (g - g_min) / span))
                        grad_mult = min_m + (max_m - min_m) * (norm ** pwr)
                        # Fuse: spectral*(1-blend) + grad*blend
                        spectral_lr = self.optimizer.param_groups[a_idx]['lr']
                        spectral_mult = spectral_lr / max(base, 1e-12)
                        fused_mult = (1.0 - blend) * spectral_mult + blend * grad_mult
                        fused_by_aidx[a_idx] = float(fused_mult)

                    # Optional: mean normalization to avoid overall LR drift
                    if getattr(self.args, 'sa_normalize_mean', True) and len(fused_by_aidx) > 0:
                        mean_fused = sum(fused_by_aidx.values()) / float(len(fused_by_aidx))
                        if mean_fused > 0:
                            for a_idx in list(fused_by_aidx.keys()):
                                fused_by_aidx[a_idx] = float(fused_by_aidx[a_idx] / mean_fused)
                                fused_by_aidx[a_idx] = float(max(min_m, min(max_m, fused_by_aidx[a_idx])))

                    # Apply to groups and sync B groups
                    for a_idx, fused_mult in fused_by_aidx.items():
                        self.optimizer.param_groups[a_idx]['lr'] = base * fused_mult
                        if hasattr(self, 'sa_a_multiplier'):
                            self.sa_a_multiplier[a_idx] = fused_mult
                        for b_idx, mapped_a in getattr(self, 'dylora_B_to_A_index', {}).items():
                            if mapped_a == a_idx:
                                ratio = self.dylora_current_lr_ratio if getattr(self.args, 'use_dylora', False) else (getattr(self.args, 'loraplus_lr_ratio', 1.0) or 1.0)
                                self.optimizer.param_groups[b_idx]['lr'] = (base * fused_mult) * ratio

                    # Log statistics
                    vals = list(fused_by_aidx.values())
                    f_min, f_max = min(vals), max(vals)
                    f_mean = sum(vals) / len(vals)
                    logger.info(
                        f"[SA-LoRA] Grad calibration fused multipliers applied (step={self.state.global_step + 1}). stats -> min={f_min:.4f}, mean={f_mean:.4f}, max={f_max:.4f}"
                    )

        # 3) warmup: linearly transition from 1 to target multipliers over first sa_warmup_steps
        if getattr(self, 'sa_enabled', False) and getattr(self.args, 'sa_warmup_steps', 0) > 0:
            step = self.state.global_step + 1
            warm = max(1, int(self.args.sa_warmup_steps))
            if step <= warm:
                frac = float(step) / float(warm)
                # Linearly interpolate LR for each SA A-group index
                for a_idx in getattr(self, 'sa_a_group_indices', []):
                    base = self.optimizer.param_groups[0]['lr']
                    target_mult = getattr(self, 'sa_a_multiplier', {}).get(a_idx, 1.0)
                    cur_mult = 1.0 + (target_mult - 1.0) * frac
                    self.optimizer.param_groups[a_idx]['lr'] = base * cur_mult
                # Sync corresponding B groups (maintain ratio)
                for b_idx in getattr(self, 'dylora_groupB_indices', []):
                    a_idx = getattr(self, 'dylora_B_to_A_index', {}).get(b_idx, 0)
                    base_lr = self.optimizer.param_groups[a_idx]['lr']
                    ratio = self.dylora_current_lr_ratio if getattr(self.args, 'use_dylora', False) else (getattr(self.args, 'loraplus_lr_ratio', 1.0) or 1.0)
                    self.optimizer.param_groups[b_idx]['lr'] = base_lr * ratio

        # Do not perform optimizer.step()/scheduler.step() here; the Trainer training loop handles those
        return loss.detach() / self.args.gradient_accumulation_steps
    # --- DyLoRA END ---
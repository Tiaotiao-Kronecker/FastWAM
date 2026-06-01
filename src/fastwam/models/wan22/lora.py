from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_ACTION_LORA_TARGETS = (
    "self_attn.q",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.o",
    "ffn.2",
)


class LoRALinear(nn.Linear):
    """State-dict-compatible LoRA wrapper for an existing Linear layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        rank = int(rank)
        if rank <= 0:
            raise ValueError(f"`rank` must be positive, got {rank}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}.")
        super().__init__(
            in_features=base.in_features,
            out_features=base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        with torch.no_grad():
            self.weight.copy_(base.weight)
            if base.bias is not None:
                self.bias.copy_(base.bias)

        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.lora_rank = rank
        self.lora_alpha = float(alpha)
        self.scaling = self.lora_alpha / float(rank)
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base + self.scaling * delta


def normalize_lora_targets(target_modules: Sequence[str] | str | None) -> tuple[str, ...]:
    if target_modules is None:
        return DEFAULT_ACTION_LORA_TARGETS
    if isinstance(target_modules, str):
        value = target_modules.strip()
        if not value or value.lower() == "all":
            return DEFAULT_ACTION_LORA_TARGETS
        target_modules = [part.strip() for part in value.split(",")]
    targets = tuple(str(target).strip() for target in target_modules if str(target).strip())
    if not targets:
        return DEFAULT_ACTION_LORA_TARGETS
    return targets


def matches_lora_target(module_name: str, targets: Sequence[str]) -> bool:
    return any(module_name == target or module_name.endswith(f".{target}") for target in targets)


def _set_child_module(parent: nn.Module, child_name: str, child: nn.Module) -> None:
    if isinstance(parent, nn.Sequential) and child_name.isdigit():
        parent[int(child_name)] = child
    else:
        setattr(parent, child_name, child)


def install_lora_layers(
    root: nn.Module,
    *,
    target_modules: Sequence[str] | str | None = None,
    rank: int = 4,
    alpha: float = 4.0,
    dropout: float = 0.0,
) -> list[str]:
    targets = normalize_lora_targets(target_modules)
    installed: list[str] = []
    for module_name, module in list(root.named_modules()):
        if not module_name or not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        if not matches_lora_target(module_name, targets):
            continue
        parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
        parent = root.get_submodule(parent_name) if parent_name else root
        _set_child_module(
            parent,
            child_name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        installed.append(module_name)
    return installed


def iter_lora_parameters(root: nn.Module) -> Iterable[nn.Parameter]:
    for module in root.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def set_lora_trainable(root: nn.Module, trainable: bool) -> None:
    for module in root.modules():
        if not isinstance(module, LoRALinear):
            continue
        module.weight.requires_grad_(False)
        if module.bias is not None:
            module.bias.requires_grad_(False)
        module.lora_A.requires_grad_(trainable)
        module.lora_B.requires_grad_(trainable)
        module.train(trainable)


def lora_parameter_count(root: nn.Module) -> int:
    return sum(param.numel() for param in iter_lora_parameters(root))

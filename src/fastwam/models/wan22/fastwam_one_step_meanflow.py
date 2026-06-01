from typing import Optional

import torch
import torch.nn as nn

from .fastwam_one_step_action import FastWAMOneStepAction
from .lora import install_lora_layers, iter_lora_parameters, lora_parameter_count, set_lora_trainable
from .wan_video_dit import force_manual_attention, sinusoidal_embedding_1d


class FastWAMOneStepMeanFlow(FastWAMOneStepAction):
    """Action-only mean-flow fine-tuning variant for one-step action generation."""

    @classmethod
    def from_wan22_pretrained(
        cls,
        *,
        meanflow_start_timestep: Optional[float] = 0.0,
        meanflow_end_timestep: Optional[float] = None,
        meanflow_derivative_epsilon: float = 0.05,
        meanflow_objective: str = "paper_jvp",
        meanflow_random_timesteps: bool = True,
        meanflow_equal_time_prob: float = 0.25,
        meanflow_trainable_scope: str = "action",
        meanflow_train_proprio_encoder: bool = True,
        meanflow_conditioner_mode: str = "additive_start",
        loss_lambda_meanflow_target: float = 0.5,
        loss_lambda_equal_time_velocity: float = 0.0,
        loss_lambda_action_velocity: float = 0.25,
        loss_lambda_action_endpoint: float = 0.25,
        meanflow_equal_time_anchor_prob: float = 0.0,
        meanflow_lora_rank: int = 0,
        meanflow_lora_alpha: float = 1.0,
        meanflow_lora_dropout: float = 0.0,
        meanflow_lora_target_modules: Optional[list[str] | str] = None,
        freeze_video_expert: bool = True,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(
            one_step_action_timestep=meanflow_end_timestep,
            loss_lambda_action_velocity=loss_lambda_action_velocity,
            loss_lambda_action_endpoint=loss_lambda_action_endpoint,
            freeze_video_expert=freeze_video_expert,
            **kwargs,
        )
        model.meanflow_start_timestep = meanflow_start_timestep
        model.meanflow_derivative_epsilon = float(meanflow_derivative_epsilon)
        model.meanflow_objective = str(meanflow_objective).strip().lower()
        model.meanflow_random_timesteps = bool(meanflow_random_timesteps)
        model.meanflow_equal_time_prob = float(meanflow_equal_time_prob)
        model.meanflow_trainable_scope = str(meanflow_trainable_scope).strip().lower()
        model.meanflow_train_proprio_encoder = bool(meanflow_train_proprio_encoder)
        model.meanflow_conditioner_mode = str(meanflow_conditioner_mode).strip().lower()
        model.loss_lambda_meanflow_target = float(loss_lambda_meanflow_target)
        model.loss_lambda_equal_time_velocity = float(loss_lambda_equal_time_velocity)
        model.meanflow_equal_time_anchor_prob = float(meanflow_equal_time_anchor_prob)
        model.meanflow_lora_rank = int(meanflow_lora_rank)
        model.meanflow_lora_alpha = float(meanflow_lora_alpha)
        model.meanflow_lora_dropout = float(meanflow_lora_dropout)
        model.meanflow_lora_target_modules = meanflow_lora_target_modules
        model.meanflow_lora_installed_modules = []
        if model.meanflow_lora_rank > 0:
            model._install_action_lora()
        model._install_meanflow_start_conditioner()
        return model

    def _install_action_lora(self) -> None:
        if getattr(self, "meanflow_lora_rank", 0) <= 0:
            return
        installed = install_lora_layers(
            self.action_expert,
            target_modules=getattr(self, "meanflow_lora_target_modules", None),
            rank=int(getattr(self, "meanflow_lora_rank", 4)),
            alpha=float(getattr(self, "meanflow_lora_alpha", 4.0)),
            dropout=float(getattr(self, "meanflow_lora_dropout", 0.0)),
        )
        if installed:
            self.meanflow_lora_installed_modules = installed
        set_lora_trainable(self.action_expert, False)

    def _set_action_lora_trainable(self, trainable: bool) -> None:
        if getattr(self, "meanflow_lora_rank", 0) <= 0:
            return
        self._install_action_lora()
        set_lora_trainable(self.action_expert, trainable)

    def _meanflow_conditioner_mode(self) -> str:
        mode = str(getattr(self, "meanflow_conditioner_mode", "additive_start")).strip().lower()
        aliases = {
            "additive": "additive_start",
            "start": "additive_start",
            "start_add": "additive_start",
            "joint": "joint_delta",
            "joint_time": "joint_delta",
        }
        return aliases.get(mode, mode)

    def _install_meanflow_start_conditioner(self) -> None:
        mode = self._meanflow_conditioner_mode()
        hidden_dim = int(self.action_expert.hidden_dim)
        freq_dim = int(self.action_expert.freq_dim)

        if mode == "additive_start":
            if hasattr(self.action_expert, "meanflow_start_embedding"):
                return
            self.action_expert.meanflow_start_embedding = nn.Sequential(
                nn.Linear(freq_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.action_expert.meanflow_start_projection = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim * 6),
            )
            final = self.action_expert.meanflow_start_projection[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            self.action_expert.meanflow_start_embedding.to(device=self.device, dtype=self.torch_dtype)
            self.action_expert.meanflow_start_projection.to(device=self.device, dtype=self.torch_dtype)
            return

        if mode == "joint_delta":
            if hasattr(self.action_expert, "meanflow_joint_time_embedding"):
                return
            self.action_expert.meanflow_joint_time_embedding = nn.Sequential(
                nn.Linear(freq_dim * 3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.action_expert.meanflow_joint_time_projection = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim * 6),
            )
            final = self.action_expert.meanflow_joint_time_projection[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            self.action_expert.meanflow_joint_time_embedding.to(device=self.device, dtype=self.torch_dtype)
            self.action_expert.meanflow_joint_time_projection.to(device=self.device, dtype=self.torch_dtype)
            return

        raise ValueError(
            "`meanflow_conditioner_mode` must be one of ['additive_start', 'joint_delta'], "
            f"got {mode!r}."
        )

    def _meanflow_conditioner_modules(self) -> tuple[nn.Module, ...]:
        self._install_meanflow_start_conditioner()
        mode = self._meanflow_conditioner_mode()
        if mode == "additive_start":
            return (
                self.action_expert.meanflow_start_embedding,
                self.action_expert.meanflow_start_projection,
            )
        if mode == "joint_delta":
            return (
                self.action_expert.meanflow_joint_time_embedding,
                self.action_expert.meanflow_joint_time_projection,
            )
        raise ValueError(
            "`meanflow_conditioner_mode` must be one of ['additive_start', 'joint_delta'], "
            f"got {mode!r}."
        )

    def _build_meanflow_start_timestep(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        timestep = self.meanflow_start_timestep
        if timestep is None:
            timestep = 0.0
        timestep = float(timestep)
        max_timestep = float(self.train_action_scheduler.num_train_timesteps)
        if timestep < 0.0 or timestep >= max_timestep:
            raise ValueError(
                "`meanflow_start_timestep` must be in [0, action_num_train_timesteps), "
                f"got {timestep}."
            )
        return torch.full((batch_size,), timestep, device=self.device, dtype=dtype)

    def _interval_view(self, timestep_start: torch.Tensor, timestep_end: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        interval = (timestep_end - timestep_start) / float(self.train_action_scheduler.num_train_timesteps)
        return interval.to(device=sample.device, dtype=sample.dtype).view(
            sample.shape[0],
            *([1] * (sample.ndim - 1)),
        )

    def _sigma_view(self, sigma: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        return sigma.to(device=sample.device, dtype=sample.dtype).view(
            sample.shape[0],
            *([1] * (sample.ndim - 1)),
        )

    def _set_meanflow_conditioner_trainable(self, trainable: bool) -> None:
        for module in self._meanflow_conditioner_modules():
            module.train(trainable)
            module.requires_grad_(trainable)

    def configure_trainable_parameters(self):
        super().configure_trainable_parameters()
        self._install_meanflow_start_conditioner()
        self._install_action_lora()

        if not getattr(self, "meanflow_train_proprio_encoder", True):
            proprio_encoder = getattr(self, "proprio_encoder", None)
            if proprio_encoder is not None:
                proprio_encoder.eval()
                proprio_encoder.requires_grad_(False)

        scope = getattr(self, "meanflow_trainable_scope", "action")
        if scope == "action":
            self.action_expert.train()
            self.action_expert.requires_grad_(True)
            self._set_meanflow_conditioner_trainable(True)
            self._set_action_lora_trainable(True)
        elif scope == "conditioner":
            self.action_expert.eval()
            self.action_expert.requires_grad_(False)
            self._set_meanflow_conditioner_trainable(True)
            self._set_action_lora_trainable(False)
        elif scope == "conditioner_head":
            self.action_expert.eval()
            self.action_expert.requires_grad_(False)
            self._set_meanflow_conditioner_trainable(True)
            self.action_expert.head.train()
            self.action_expert.head.requires_grad_(True)
            self._set_action_lora_trainable(False)
        elif scope == "conditioner_head_lora":
            self.action_expert.eval()
            self.action_expert.requires_grad_(False)
            self._set_meanflow_conditioner_trainable(True)
            self.action_expert.head.train()
            self.action_expert.head.requires_grad_(True)
            self._set_action_lora_trainable(True)
        else:
            raise ValueError(
                "`meanflow_trainable_scope` must be one of "
                "['action', 'conditioner', 'conditioner_head', 'conditioner_head_lora'], "
                f"got {scope!r}."
            )

    def extra_trainable_parameters(self):
        for module in self._meanflow_conditioner_modules():
            yield from module.parameters()
        scope = getattr(self, "meanflow_trainable_scope", "action")
        if scope in ("conditioner_head", "conditioner_head_lora"):
            yield from self.action_expert.head.parameters()
        if scope == "conditioner_head_lora":
            yield from iter_lora_parameters(self.action_expert)

    def trainable_parameter_summary(self) -> dict[str, int]:
        conditioner = sum(param.numel() for module in self._meanflow_conditioner_modules() for param in module.parameters() if param.requires_grad)
        head = sum(param.numel() for param in self.action_expert.head.parameters() if param.requires_grad)
        lora = sum(param.numel() for param in iter_lora_parameters(self.action_expert) if param.requires_grad)
        total = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {
            "trainable_total": int(total),
            "trainable_conditioner": int(conditioner),
            "trainable_head": int(head),
            "trainable_lora": int(lora),
            "lora_total": int(lora_parameter_count(self.action_expert)),
        }

    def _apply_meanflow_start_conditioning(
        self,
        action_pre: dict,
        timestep_start: torch.Tensor,
        timestep_action: Optional[torch.Tensor] = None,
    ) -> dict:
        self._install_meanflow_start_conditioner()
        if timestep_start.ndim != 1:
            raise ValueError(
                f"`timestep_start` must be 1D [B], got shape {tuple(timestep_start.shape)}"
            )
        start_position = timestep_start.to(
            device=action_pre["t_mod"].device,
            dtype=action_pre["t_mod"].dtype,
        )
        mode = self._meanflow_conditioner_mode()

        if mode == "additive_start":
            start_emb = sinusoidal_embedding_1d(self.action_expert.freq_dim, start_position)
            start_emb = start_emb.to(device=action_pre["t_mod"].device, dtype=action_pre["t_mod"].dtype)
            start_hidden = self.action_expert.meanflow_start_embedding(start_emb)
            start_mod = self.action_expert.meanflow_start_projection(start_hidden).unflatten(
                1,
                (6, self.action_expert.hidden_dim),
            )
            action_pre = dict(action_pre)
            action_pre["t_mod"] = action_pre["t_mod"] + start_mod
            return action_pre

        if mode == "joint_delta":
            if timestep_action is None:
                raise ValueError("`timestep_action` is required when `meanflow_conditioner_mode='joint_delta'`.")
            if timestep_action.ndim != 1:
                raise ValueError(
                    f"`timestep_action` must be 1D [B], got shape {tuple(timestep_action.shape)}"
                )
            action_position = timestep_action.to(
                device=action_pre["t_mod"].device,
                dtype=action_pre["t_mod"].dtype,
            )
            interval_position = action_position - start_position
            action_emb = sinusoidal_embedding_1d(self.action_expert.freq_dim, action_position)
            start_emb = sinusoidal_embedding_1d(self.action_expert.freq_dim, start_position)
            interval_emb = sinusoidal_embedding_1d(self.action_expert.freq_dim, interval_position)
            joint_emb = torch.cat([action_emb, start_emb, interval_emb], dim=1).to(
                device=action_pre["t_mod"].device,
                dtype=action_pre["t_mod"].dtype,
            )
            joint_hidden = self.action_expert.meanflow_joint_time_embedding(joint_emb)
            joint_mod = self.action_expert.meanflow_joint_time_projection(joint_hidden).unflatten(
                1,
                (6, self.action_expert.hidden_dim),
            )
            action_pre = dict(action_pre)
            action_pre["t_mod"] = action_pre["t_mod"] + joint_mod
            return action_pre

        raise ValueError(
            "`meanflow_conditioner_mode` must be one of ['additive_start', 'joint_delta'], "
            f"got {mode!r}."
        )

    def _predict_meanflow_action_velocity(
        self,
        *,
        video_pre: dict,
        action_tokens: torch.Tensor,
        timestep_action: torch.Tensor,
        timestep_start: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self._apply_meanflow_start_conditioning(
            action_pre=action_pre,
            timestep_start=timestep_start,
            timestep_action=timestep_action,
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        return self.action_expert.post_dit(tokens_out["action"], action_pre)

    def _sample_meanflow_sigma_pair(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        equal_time_prob: Optional[float] = None,
        min_interval: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = float(self.train_action_scheduler.num_train_timesteps)
        if not getattr(self, "meanflow_random_timesteps", True):
            timestep_end = self._build_one_step_action_timestep(batch_size=batch_size, dtype=dtype)
            timestep_start = self._build_meanflow_start_timestep(batch_size=batch_size, dtype=dtype)
            sigma_end = (timestep_end / steps).to(device=self.device, dtype=dtype)
            sigma_start = (timestep_start / steps).to(device=self.device, dtype=dtype)
            return sigma_start, sigma_end

        first = torch.rand((batch_size,), device=self.device, dtype=torch.float32)
        second = torch.rand((batch_size,), device=self.device, dtype=torch.float32)
        sigma_start = torch.minimum(first, second)
        sigma_end = torch.maximum(first, second)
        min_interval = float(min_interval)
        if min_interval < 0.0 or min_interval >= 1.0:
            raise ValueError(f"`min_interval` must be in [0, 1), got {min_interval}.")
        if min_interval > 0.0:
            interval = min_interval + (1.0 - min_interval) * (sigma_end - sigma_start)
            sigma_start = torch.minimum(sigma_start, 1.0 - interval)
            sigma_end = sigma_start + interval

        if equal_time_prob is None:
            equal_time_prob = float(getattr(self, "meanflow_equal_time_prob", 0.0))
        else:
            equal_time_prob = float(equal_time_prob)
        if equal_time_prob < 0.0 or equal_time_prob > 1.0:
            raise ValueError(f"`meanflow_equal_time_prob` must be in [0, 1], got {equal_time_prob}.")
        if equal_time_prob > 0.0:
            equal_mask = torch.rand((batch_size,), device=self.device) < equal_time_prob
            sigma_start = torch.where(equal_mask, sigma_end, sigma_start)

        return sigma_start.to(dtype=dtype), sigma_end.to(dtype=dtype)

    def _build_video_pre(
        self,
        *,
        input_latents: torch.Tensor,
        first_frame_latents: Optional[torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> dict:
        batch_size = input_latents.shape[0]
        if first_frame_latents is None:
            first_frame_latents = input_latents[:, :, 0:1]
        timestep_video = torch.zeros(
            (batch_size,),
            device=self.device,
            dtype=input_latents.dtype,
        )
        return self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )

    def _maybe_equal_time_anchor_loss(
        self,
        *,
        video_pre: dict,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], dict[str, float]]:
        lambda_anchor = float(getattr(self, "loss_lambda_equal_time_velocity", 0.0))
        anchor_prob = float(getattr(self, "meanflow_equal_time_anchor_prob", 0.0))
        if lambda_anchor == 0.0 or anchor_prob <= 0.0:
            return None, {"equal_time_anchor_applied": 0.0}
        if anchor_prob > 1.0:
            raise ValueError(f"`meanflow_equal_time_anchor_prob` must be in [0, 1], got {anchor_prob}.")
        anchor_decision = torch.rand((), device=action.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(anchor_decision, src=0)
        if anchor_decision >= anchor_prob:
            return None, {"equal_time_anchor_applied": 0.0}

        batch_size = action.shape[0]
        sigma = torch.rand((batch_size,), device=action.device, dtype=torch.float32)
        timestep = (sigma * float(self.train_action_scheduler.num_train_timesteps)).to(dtype=action.dtype)
        noise_action = torch.randn_like(action)
        noisy_action = (1.0 - self._sigma_view(sigma, action)) * action + self._sigma_view(
            sigma,
            action,
        ) * noise_action
        target_velocity = self.train_action_scheduler.training_target(
            action,
            noise_action,
            timestep,
        )
        pred_velocity = self._predict_meanflow_action_velocity(
            video_pre=video_pre,
            action_tokens=noisy_action,
            timestep_action=timestep,
            timestep_start=timestep,
            context=context,
            context_mask=context_mask,
        )
        loss_anchor = self._masked_action_mse(
            pred=pred_velocity,
            target=target_velocity,
            action_is_pad=action_is_pad,
        )
        return loss_anchor, {
            "equal_time_anchor_applied": 1.0,
            "equal_time_anchor_sigma": float(sigma.detach().float().mean().item()),
        }

    def _set_checkpointing_enabled(self, enabled: bool) -> dict[str, bool]:
        states = {
            "mot_checkpoint_mixed_attn": bool(getattr(self.mot, "mot_checkpoint_mixed_attn", False)),
            "video_use_gradient_checkpointing": bool(getattr(self.video_expert, "use_gradient_checkpointing", False)),
            "action_use_gradient_checkpointing": bool(getattr(self.action_expert, "use_gradient_checkpointing", False)),
        }
        self.mot.mot_checkpoint_mixed_attn = bool(enabled) and states["mot_checkpoint_mixed_attn"]
        if hasattr(self.video_expert, "use_gradient_checkpointing"):
            self.video_expert.use_gradient_checkpointing = bool(enabled) and states["video_use_gradient_checkpointing"]
        if hasattr(self.action_expert, "use_gradient_checkpointing"):
            self.action_expert.use_gradient_checkpointing = bool(enabled) and states["action_use_gradient_checkpointing"]
        return states

    def _restore_checkpointing_state(self, states: dict[str, bool]) -> None:
        self.mot.mot_checkpoint_mixed_attn = states["mot_checkpoint_mixed_attn"]
        if hasattr(self.video_expert, "use_gradient_checkpointing"):
            self.video_expert.use_gradient_checkpointing = states["video_use_gradient_checkpointing"]
        if hasattr(self.action_expert, "use_gradient_checkpointing"):
            self.action_expert.use_gradient_checkpointing = states["action_use_gradient_checkpointing"]

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        timestep_start = self._build_meanflow_start_timestep(
            batch_size=latents_action.shape[0],
            dtype=latents_action.dtype,
        )
        action_pre = self._apply_meanflow_start_conditioning(
            action_pre=action_pre,
            timestep_start=timestep_start,
            timestep_action=timestep_action,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    def _training_loss_paper_jvp(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]
        jvp_dtype = action.dtype

        video_pre = self._build_video_pre(
            input_latents=input_latents,
            first_frame_latents=inputs["first_frame_latents"],
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )

        noise_action = torch.randn_like(action)
        sigma_start, sigma_end = self._sample_meanflow_sigma_pair(
            batch_size=batch_size,
            dtype=action.dtype,
        )
        timestep_start = sigma_start * float(self.train_action_scheduler.num_train_timesteps)
        timestep_end = sigma_end * float(self.train_action_scheduler.num_train_timesteps)
        timestep_start = timestep_start.to(dtype=action.dtype)
        timestep_end = timestep_end.to(dtype=action.dtype)
        noisy_action = (1.0 - self._sigma_view(sigma_end, action)) * action + self._sigma_view(
            sigma_end,
            action,
        ) * noise_action
        noisy_action = noisy_action.to(dtype=jvp_dtype)
        target_action_velocity = self.train_action_scheduler.training_target(
            action,
            noise_action,
            timestep_end,
        ).to(dtype=jvp_dtype)

        def u_fn(action_tokens: torch.Tensor, sigma_t: torch.Tensor, sigma_r: torch.Tensor) -> torch.Tensor:
            action_tokens = action_tokens.to(dtype=jvp_dtype)
            sigma_t = sigma_t.to(dtype=jvp_dtype)
            sigma_r = sigma_r.to(dtype=jvp_dtype)
            timestep_action = (sigma_t * float(self.train_action_scheduler.num_train_timesteps)).to(dtype=jvp_dtype)
            timestep_start = (sigma_r * float(self.train_action_scheduler.num_train_timesteps)).to(dtype=jvp_dtype)
            return self._predict_meanflow_action_velocity(
                video_pre=video_pre,
                action_tokens=action_tokens,
                timestep_action=timestep_action,
                timestep_start=timestep_start,
                context=context,
                context_mask=context_mask,
            )

        checkpoint_states = self._set_checkpointing_enabled(False)
        try:
            with force_manual_attention(True):
                pred_mean_velocity, dudt = torch.func.jvp(
                    u_fn,
                    (noisy_action, sigma_end, sigma_start),
                    (
                        target_action_velocity,
                        torch.ones_like(sigma_end),
                        torch.zeros_like(sigma_start),
                    ),
                )
        finally:
            self._restore_checkpointing_state(checkpoint_states)
        meanflow_target = target_action_velocity - self._sigma_view(
            sigma_end - sigma_start,
            action,
        ) * dudt
        loss_meanflow_target = self._masked_action_mse(
            pred=pred_mean_velocity,
            target=meanflow_target.detach(),
            action_is_pad=action_is_pad,
        )

        loss_total = self.loss_lambda_meanflow_target * loss_meanflow_target
        loss_dict = {
            "loss_meanflow_target": self.loss_lambda_meanflow_target
            * float(loss_meanflow_target.detach().item()),
            "meanflow_sigma_start": float(sigma_start.detach().float().mean().item()),
            "meanflow_sigma_end": float(sigma_end.detach().float().mean().item()),
            "meanflow_interval": float((sigma_end - sigma_start).detach().float().mean().item()),
            "meanflow_dudt_rms": float(dudt.detach().float().pow(2).mean().sqrt().item()),
            "pred_mean_velocity_rms": float(pred_mean_velocity.detach().float().pow(2).mean().sqrt().item()),
            "target_meanflow_rms": float(meanflow_target.detach().float().pow(2).mean().sqrt().item()),
        }

        loss_equal_time_velocity, anchor_metrics = self._maybe_equal_time_anchor_loss(
            video_pre=video_pre,
            action=action,
            action_is_pad=action_is_pad,
            context=context,
            context_mask=context_mask,
        )
        loss_dict.update(anchor_metrics)
        if loss_equal_time_velocity is not None:
            loss_total = loss_total + self.loss_lambda_equal_time_velocity * loss_equal_time_velocity
            loss_dict["loss_equal_time_velocity"] = self.loss_lambda_equal_time_velocity * float(
                loss_equal_time_velocity.detach().item()
            )

        if self.loss_lambda_action_velocity != 0.0:
            loss_action_velocity = self._masked_action_mse(
                pred=pred_mean_velocity,
                target=target_action_velocity,
                action_is_pad=action_is_pad,
            )
            loss_total = loss_total + self.loss_lambda_action_velocity * loss_action_velocity
            loss_dict["loss_action_velocity"] = self.loss_lambda_action_velocity * float(
                loss_action_velocity.detach().item()
            )

        if self.loss_lambda_action_endpoint != 0.0:
            pred_action_endpoint = noisy_action - self._sigma_view(
                sigma_end - sigma_start,
                action,
            ) * pred_mean_velocity
            loss_action_endpoint = self._masked_action_mse(
                pred=pred_action_endpoint,
                target=action,
                action_is_pad=action_is_pad,
            )
            loss_total = loss_total + self.loss_lambda_action_endpoint * loss_action_endpoint
            loss_dict["loss_action_endpoint"] = self.loss_lambda_action_endpoint * float(
                loss_action_endpoint.detach().item()
            )

        return loss_total, loss_dict

    def _training_loss_finite_difference(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        video_pre = self._build_video_pre(
            input_latents=input_latents,
            first_frame_latents=inputs["first_frame_latents"],
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )

        noise_action = torch.randn_like(action)
        sigma_start, sigma_end = self._sample_meanflow_sigma_pair(
            batch_size=batch_size,
            dtype=torch.float32,
            equal_time_prob=0.0,
            min_interval=float(self.meanflow_derivative_epsilon),
        )
        timestep_start = sigma_start * float(self.train_action_scheduler.num_train_timesteps)
        timestep_end = sigma_end * float(self.train_action_scheduler.num_train_timesteps)
        if torch.any(timestep_start >= timestep_end):
            raise ValueError("Mean-flow start timestep must be smaller than end timestep.")

        noisy_action = (1.0 - self._sigma_view(sigma_end, action)) * action + self._sigma_view(
            sigma_end,
            action,
        ) * noise_action
        target_action_velocity = self.train_action_scheduler.training_target(
            action,
            noise_action,
            timestep_end,
        )
        pred_mean_velocity = self._predict_meanflow_action_velocity(
            video_pre=video_pre,
            action_tokens=noisy_action,
            timestep_action=timestep_end,
            timestep_start=timestep_start,
            context=context,
            context_mask=context_mask,
        )

        eps = float(self.meanflow_derivative_epsilon)
        if eps <= 0.0 or eps >= 1.0:
            raise ValueError(f"`meanflow_derivative_epsilon` must be in (0, 1), got {eps}.")
        eps_timestep = eps * float(self.train_action_scheduler.num_train_timesteps)
        prev_timestep = torch.maximum(timestep_start, timestep_end - eps_timestep)
        eps_actual = ((timestep_end - prev_timestep) / float(self.train_action_scheduler.num_train_timesteps)).to(
            device=action.device,
            dtype=action.dtype,
        )
        if torch.any(eps_actual <= 0):
            raise ValueError("Mean-flow finite-difference epsilon collapsed to zero.")
        prev_action = noisy_action - eps_actual.view(batch_size, *([1] * (action.ndim - 1))) * target_action_velocity
        with torch.no_grad():
            pred_prev_mean_velocity = self._predict_meanflow_action_velocity(
                video_pre=video_pre,
                action_tokens=prev_action,
                timestep_action=prev_timestep,
                timestep_start=timestep_start,
                context=context,
                context_mask=context_mask,
            )
            dudt = (
                pred_mean_velocity.detach() - pred_prev_mean_velocity
            ) / eps_actual.view(batch_size, *([1] * (action.ndim - 1)))
            meanflow_target = target_action_velocity - self._interval_view(
                timestep_start,
                timestep_end,
                action,
            ) * dudt

        interval = self._interval_view(timestep_start, timestep_end, action)
        pred_action_endpoint = noisy_action - interval * pred_mean_velocity

        loss_meanflow_target = self._masked_action_mse(
            pred=pred_mean_velocity,
            target=meanflow_target,
            action_is_pad=action_is_pad,
        )
        loss_action_velocity = self._masked_action_mse(
            pred=pred_mean_velocity,
            target=target_action_velocity,
            action_is_pad=action_is_pad,
        )
        loss_action_endpoint = self._masked_action_mse(
            pred=pred_action_endpoint,
            target=action,
            action_is_pad=action_is_pad,
        )

        loss_total = (
            self.loss_lambda_meanflow_target * loss_meanflow_target
            + self.loss_lambda_action_velocity * loss_action_velocity
            + self.loss_lambda_action_endpoint * loss_action_endpoint
        )
        loss_dict = {
            "loss_meanflow_target": self.loss_lambda_meanflow_target
            * float(loss_meanflow_target.detach().item()),
            "loss_action_velocity": self.loss_lambda_action_velocity
            * float(loss_action_velocity.detach().item()),
            "loss_action_endpoint": self.loss_lambda_action_endpoint
            * float(loss_action_endpoint.detach().item()),
            "meanflow_sigma_start": float(sigma_start.detach().float().mean().item()),
            "meanflow_sigma_end": float(sigma_end.detach().float().mean().item()),
            "meanflow_interval": float((sigma_end - sigma_start).detach().float().mean().item()),
            "meanflow_dudt_rms": float(dudt.detach().float().pow(2).mean().sqrt().item()),
            "pred_mean_velocity_rms": float(pred_mean_velocity.detach().float().pow(2).mean().sqrt().item()),
            "target_meanflow_rms": float(meanflow_target.detach().float().pow(2).mean().sqrt().item()),
        }
        loss_equal_time_velocity, anchor_metrics = self._maybe_equal_time_anchor_loss(
            video_pre=video_pre,
            action=action,
            action_is_pad=action_is_pad,
            context=context,
            context_mask=context_mask,
        )
        loss_dict.update(anchor_metrics)
        if loss_equal_time_velocity is not None:
            loss_total = loss_total + self.loss_lambda_equal_time_velocity * loss_equal_time_velocity
            loss_dict["loss_equal_time_velocity"] = self.loss_lambda_equal_time_velocity * float(
                loss_equal_time_velocity.detach().item()
            )
        return loss_total, loss_dict

    def training_loss(self, sample, tiled: bool = False):
        objective = getattr(self, "meanflow_objective", "paper_jvp")
        if objective == "paper_jvp":
            return self._training_loss_paper_jvp(sample, tiled=tiled)
        if objective == "finite_difference":
            return self._training_loss_finite_difference(sample, tiled=tiled)
        raise ValueError(
            "`meanflow_objective` must be one of ['paper_jvp', 'finite_difference'], "
            f"got {objective!r}."
        )

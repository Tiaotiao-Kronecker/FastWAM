import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fastwam import FastWAM
from .fastwam_joint import FastWAMJoint
from .wan_video_dit import force_manual_attention, sinusoidal_embedding_1d


class FastWAMCJointMeanFlow(FastWAMJoint):
    """C-joint MeanFlow variant: original video flow loss plus action MeanFlow loss."""

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
        meanflow_trainable_scope: str = "joint",
        meanflow_train_proprio_encoder: bool = True,
        loss_lambda_meanflow_action: float = 1.0,
        loss_lambda_action_velocity: float = 0.0,
        loss_lambda_action_endpoint: float = 0.0,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(**kwargs)
        model.meanflow_start_timestep = meanflow_start_timestep
        model.meanflow_end_timestep = meanflow_end_timestep
        model.meanflow_derivative_epsilon = float(meanflow_derivative_epsilon)
        model.meanflow_objective = str(meanflow_objective).strip().lower()
        model.meanflow_random_timesteps = bool(meanflow_random_timesteps)
        model.meanflow_equal_time_prob = float(meanflow_equal_time_prob)
        model.meanflow_trainable_scope = str(meanflow_trainable_scope).strip().lower()
        model.meanflow_train_proprio_encoder = bool(meanflow_train_proprio_encoder)
        model.loss_lambda_meanflow_action = float(loss_lambda_meanflow_action)
        model.loss_lambda_action_velocity = float(loss_lambda_action_velocity)
        model.loss_lambda_action_endpoint = float(loss_lambda_action_endpoint)
        model._install_meanflow_start_conditioner()
        return model

    @staticmethod
    def _masked_action_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss_token = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=2)
        if action_is_pad is None:
            return loss_token.mean()
        valid = (~action_is_pad).to(device=loss_token.device, dtype=loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return ((loss_token * valid).sum(dim=1) / valid_sum).mean()

    @staticmethod
    def _masked_action_rms(
        value: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> float:
        value = value.detach().float()
        if action_is_pad is None:
            return float(value.pow(2).mean().sqrt().item())
        valid = (~action_is_pad).to(device=value.device, dtype=value.dtype).unsqueeze(-1)
        denom = (valid.sum() * value.shape[-1]).clamp(min=1.0)
        return float((value.pow(2) * valid).sum().div(denom).sqrt().item())

    @classmethod
    def _add_debug_norm_metrics(
        cls,
        loss_dict: dict[str, float],
        *,
        action_is_pad: Optional[torch.Tensor],
        **tensors: torch.Tensor,
    ) -> None:
        if os.environ.get("FASTWAM_MEANFLOW_DEBUG_NORMS", "").strip().lower() not in {"1", "true", "yes"}:
            return
        for name, tensor in tensors.items():
            loss_dict[f"debug_{name}_rms"] = cls._masked_action_rms(tensor, action_is_pad)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        # Keep C-joint action training aligned with FastWAM's release policy:
        # action tokens may attend to action tokens and first-frame video tokens only.
        return FastWAM._build_mot_attention_mask(
            self,
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

    def _install_meanflow_start_conditioner(self) -> None:
        if hasattr(self.action_expert, "meanflow_start_embedding"):
            return
        hidden_dim = int(self.action_expert.hidden_dim)
        freq_dim = int(self.action_expert.freq_dim)
        self.action_expert.meanflow_start_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device=self.device, dtype=self.torch_dtype)
        self.action_expert.meanflow_start_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        ).to(device=self.device, dtype=self.torch_dtype)
        final = self.action_expert.meanflow_start_projection[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

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

    def _build_meanflow_end_timestep(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        timestep = self.meanflow_end_timestep
        if timestep is None:
            timestep = float(self.train_action_scheduler.num_train_timesteps)
        timestep = float(timestep)
        max_timestep = float(self.train_action_scheduler.num_train_timesteps)
        if timestep <= 0.0 or timestep > max_timestep:
            raise ValueError(
                "`meanflow_end_timestep` must be in (0, action_num_train_timesteps], "
                f"got {timestep}."
            )
        return torch.full((batch_size,), timestep, device=self.device, dtype=dtype)

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
            timestep_start = self._build_meanflow_start_timestep(batch_size=batch_size, dtype=dtype)
            timestep_end = self._build_meanflow_end_timestep(batch_size=batch_size, dtype=dtype)
            return timestep_start / steps, timestep_end / steps

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

    def _sigma_view(self, sigma: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        return sigma.to(device=sample.device, dtype=sample.dtype).view(
            sample.shape[0],
            *([1] * (sample.ndim - 1)),
        )

    def _interval_view(
        self,
        timestep_start: torch.Tensor,
        timestep_end: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        interval = (timestep_end - timestep_start) / float(self.train_action_scheduler.num_train_timesteps)
        return interval.to(device=sample.device, dtype=sample.dtype).view(
            sample.shape[0],
            *([1] * (sample.ndim - 1)),
        )

    def _apply_meanflow_start_conditioning(
        self,
        action_pre: dict,
        timestep_start: torch.Tensor,
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
        start_emb = sinusoidal_embedding_1d(self.action_expert.freq_dim, start_position)
        start_emb = start_emb.to(device=action_pre["t_mod"].device, dtype=action_pre["t_mod"].dtype)
        start_hidden = self.action_expert.meanflow_start_embedding(start_emb)
        start_mod = self.action_expert.meanflow_start_projection(start_hidden).unflatten(
            1,
            (6, self.action_expert.hidden_dim),
        )
        if action_pre["t_mod"].ndim == 4:
            start_mod = start_mod.unsqueeze(1)
        elif action_pre["t_mod"].ndim != 3:
            raise ValueError(f"Unsupported action `t_mod` shape: {tuple(action_pre['t_mod'].shape)}")
        action_pre = dict(action_pre)
        action_pre["t_mod"] = action_pre["t_mod"] + start_mod
        return action_pre

    def _run_joint_pre_states(
        self,
        *,
        video_pre: dict,
        action_pre: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    def _predict_video_and_action_mean_velocity(
        self,
        *,
        video_pre: dict,
        action_tokens: torch.Tensor,
        timestep_action: torch.Tensor,
        timestep_start: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self._apply_meanflow_start_conditioning(
            action_pre=action_pre,
            timestep_start=timestep_start,
        )
        return self._run_joint_pre_states(video_pre=video_pre, action_pre=action_pre)

    def _predict_action_mean_velocity(
        self,
        *,
        video_pre: dict,
        action_tokens: torch.Tensor,
        timestep_action: torch.Tensor,
        timestep_start: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, pred_action = self._predict_video_and_action_mean_velocity(
            video_pre=video_pre,
            action_tokens=action_tokens,
            timestep_action=timestep_action,
            timestep_start=timestep_start,
            context=context,
            context_mask=context_mask,
        )
        return pred_action

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

    def _compute_original_video_flow_loss(
        self,
        *,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        timestep_video: torch.Tensor,
        first_frame_latents: Optional[torch.Tensor],
        image_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        include_initial_video_step = first_frame_latents is None
        if first_frame_latents is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        return (loss_video_per_sample * video_weight).mean()

    def _add_optional_action_losses(
        self,
        *,
        loss_total: torch.Tensor,
        loss_dict: dict[str, float],
        pred_mean_velocity: torch.Tensor,
        target_action_velocity: torch.Tensor,
        noisy_action: torch.Tensor,
        sigma_interval: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
            pred_action_endpoint = noisy_action - self._sigma_view(sigma_interval, action) * pred_mean_velocity
            loss_action_endpoint = self._masked_action_mse(
                pred=pred_action_endpoint,
                target=action,
                action_is_pad=action_is_pad,
            )
            loss_total = loss_total + self.loss_lambda_action_endpoint * loss_action_endpoint
            loss_dict["loss_action_endpoint"] = self.loss_lambda_action_endpoint * float(
                loss_action_endpoint.detach().item()
            )

        return loss_total

    def configure_trainable_parameters(self):
        self._install_meanflow_start_conditioner()

        if not getattr(self, "meanflow_train_proprio_encoder", True):
            proprio_encoder = getattr(self, "proprio_encoder", None)
            if proprio_encoder is not None:
                proprio_encoder.eval()
                proprio_encoder.requires_grad_(False)

        scope = getattr(self, "meanflow_trainable_scope", "joint")
        if scope == "joint":
            self.video_expert.train()
            self.video_expert.requires_grad_(True)
            self.action_expert.train()
            self.action_expert.requires_grad_(True)
            self._set_meanflow_conditioner_trainable(True)
        elif scope == "action":
            self.video_expert.eval()
            self.video_expert.requires_grad_(False)
            self.action_expert.train()
            self.action_expert.requires_grad_(True)
            self._set_meanflow_conditioner_trainable(True)
        elif scope == "conditioner":
            self.video_expert.eval()
            self.video_expert.requires_grad_(False)
            self.action_expert.eval()
            self.action_expert.requires_grad_(False)
            self._set_meanflow_conditioner_trainable(True)
        else:
            raise ValueError(
                "`meanflow_trainable_scope` must be one of ['joint', 'action', 'conditioner'], "
                f"got {scope!r}."
            )

    def _set_meanflow_conditioner_trainable(self, trainable: bool) -> None:
        self._install_meanflow_start_conditioner()
        modules = (
            self.action_expert.meanflow_start_embedding,
            self.action_expert.meanflow_start_projection,
        )
        for module in modules:
            module.train(trainable)
            module.requires_grad_(trainable)

    def extra_trainable_parameters(self):
        self._install_meanflow_start_conditioner()
        yield from self.action_expert.meanflow_start_embedding.parameters()
        yield from self.action_expert.meanflow_start_projection.parameters()

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
            dtype=timestep_action.dtype,
        )
        action_pre = self._apply_meanflow_start_conditioning(
            action_pre=action_pre,
            timestep_start=timestep_start,
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

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, torch.Tensor]:
        return FastWAM.infer_action(
            self,
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        timestep_start = self._build_meanflow_start_timestep(
            batch_size=latents_action.shape[0],
            dtype=timestep_action.dtype,
        )
        return self._predict_video_and_action_mean_velocity(
            video_pre=video_pre,
            action_tokens=latents_action,
            timestep_action=timestep_action,
            timestep_start=timestep_start,
            context=context,
            context_mask=context_mask,
        )

    def _build_joint_training_states(self, sample, tiled: bool):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        action = inputs["action"]
        batch_size = input_latents.shape[0]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        noisy_video = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            noisy_video[:, :, 0:1] = inputs["first_frame_latents"]

        video_pre = self.video_expert.pre_dit(
            x=noisy_video,
            timestep=timestep_video,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        return inputs, video_pre, timestep_video, target_video

    def _training_loss_paper_jvp(self, sample, tiled: bool = False):
        inputs, video_pre, timestep_video, target_video = self._build_joint_training_states(sample, tiled=tiled)
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        batch_size = action.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        jvp_dtype = action.dtype

        noise_action = torch.randn_like(action)
        sigma_start, sigma_end = self._sample_meanflow_sigma_pair(
            batch_size=batch_size,
            dtype=action.dtype,
        )
        timestep_start = (sigma_start * float(self.train_action_scheduler.num_train_timesteps)).to(dtype=action.dtype)
        timestep_end = (sigma_end * float(self.train_action_scheduler.num_train_timesteps)).to(dtype=action.dtype)
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_end).to(dtype=jvp_dtype)
        target_action_velocity = self.train_action_scheduler.training_target(
            action,
            noise_action,
            timestep_end,
        ).to(dtype=jvp_dtype)

        def u_fn(action_tokens: torch.Tensor, sigma_t: torch.Tensor, sigma_r: torch.Tensor) -> torch.Tensor:
            action_tokens = action_tokens.to(dtype=jvp_dtype)
            sigma_t = sigma_t.to(dtype=jvp_dtype)
            sigma_r = sigma_r.to(dtype=jvp_dtype)
            action_timestep = (sigma_t * float(self.train_action_scheduler.num_train_timesteps)).to(dtype=jvp_dtype)
            start_timestep = (sigma_r * float(self.train_action_scheduler.num_train_timesteps)).to(dtype=jvp_dtype)
            return self._predict_action_mean_velocity(
                video_pre=video_pre,
                action_tokens=action_tokens,
                timestep_action=action_timestep,
                timestep_start=start_timestep,
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

        pred_video, _ = self._predict_video_and_action_mean_velocity(
            video_pre=video_pre,
            action_tokens=noisy_action,
            timestep_action=timestep_end,
            timestep_start=timestep_start,
            context=context,
            context_mask=context_mask,
        )

        meanflow_target = target_action_velocity - self._sigma_view(
            sigma_end - sigma_start,
            action,
        ) * dudt
        loss_video = self._compute_original_video_flow_loss(
            pred_video=pred_video,
            target_video=target_video,
            timestep_video=timestep_video,
            first_frame_latents=inputs["first_frame_latents"],
            image_is_pad=inputs["image_is_pad"],
        )
        loss_action = self._masked_action_mse(
            pred=pred_mean_velocity,
            target=meanflow_target.detach(),
            action_is_pad=action_is_pad,
        )

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_meanflow_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_meanflow_action": self.loss_lambda_meanflow_action * float(loss_action.detach().item()),
            "meanflow_sigma_start": float(sigma_start.detach().float().mean().item()),
            "meanflow_sigma_end": float(sigma_end.detach().float().mean().item()),
            "meanflow_interval": float((sigma_end - sigma_start).detach().float().mean().item()),
        }
        self._add_debug_norm_metrics(
            loss_dict,
            action_is_pad=action_is_pad,
            pred_mean_velocity=pred_mean_velocity,
            target_action_velocity=target_action_velocity,
            dudt=dudt,
            meanflow_target=meanflow_target,
        )
        return self._add_optional_action_losses(
            loss_total=loss_total,
            loss_dict=loss_dict,
            pred_mean_velocity=pred_mean_velocity,
            target_action_velocity=target_action_velocity,
            noisy_action=noisy_action,
            sigma_interval=sigma_end - sigma_start,
            action=action,
            action_is_pad=action_is_pad,
        ), loss_dict

    def _training_loss_finite_difference(self, sample, tiled: bool = False):
        inputs, video_pre, timestep_video, target_video = self._build_joint_training_states(sample, tiled=tiled)
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        batch_size = action.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]

        noise_action = torch.randn_like(action)
        sigma_start, sigma_end = self._sample_meanflow_sigma_pair(
            batch_size=batch_size,
            dtype=torch.float32,
            equal_time_prob=0.0,
            min_interval=float(self.meanflow_derivative_epsilon),
        )
        timestep_start = sigma_start * float(self.train_action_scheduler.num_train_timesteps)
        timestep_end = sigma_end * float(self.train_action_scheduler.num_train_timesteps)
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_end)
        target_action_velocity = self.train_action_scheduler.training_target(
            action,
            noise_action,
            timestep_end,
        )

        pred_video, pred_mean_velocity = self._predict_video_and_action_mean_velocity(
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
            raise ValueError("MeanFlow finite-difference epsilon collapsed to zero.")
        prev_action = noisy_action - self._sigma_view(eps_actual, action) * target_action_velocity

        with torch.no_grad():
            pred_prev_mean_velocity = self._predict_action_mean_velocity(
                video_pre=video_pre,
                action_tokens=prev_action,
                timestep_action=prev_timestep,
                timestep_start=timestep_start,
                context=context,
                context_mask=context_mask,
            )
            dudt = (pred_mean_velocity.detach() - pred_prev_mean_velocity) / self._sigma_view(
                eps_actual,
                action,
            )
            meanflow_target = target_action_velocity - self._sigma_view(
                sigma_end - sigma_start,
                action,
            ) * dudt

        loss_video = self._compute_original_video_flow_loss(
            pred_video=pred_video,
            target_video=target_video,
            timestep_video=timestep_video,
            first_frame_latents=inputs["first_frame_latents"],
            image_is_pad=inputs["image_is_pad"],
        )
        loss_action = self._masked_action_mse(
            pred=pred_mean_velocity,
            target=meanflow_target,
            action_is_pad=action_is_pad,
        )

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_meanflow_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_meanflow_action": self.loss_lambda_meanflow_action * float(loss_action.detach().item()),
            "meanflow_sigma_start": float(sigma_start.detach().float().mean().item()),
            "meanflow_sigma_end": float(sigma_end.detach().float().mean().item()),
            "meanflow_interval": float((sigma_end - sigma_start).detach().float().mean().item()),
        }
        self._add_debug_norm_metrics(
            loss_dict,
            action_is_pad=action_is_pad,
            pred_mean_velocity=pred_mean_velocity,
            target_action_velocity=target_action_velocity,
            dudt=dudt,
            meanflow_target=meanflow_target,
        )
        return self._add_optional_action_losses(
            loss_total=loss_total,
            loss_dict=loss_dict,
            pred_mean_velocity=pred_mean_velocity,
            target_action_velocity=target_action_velocity,
            noisy_action=noisy_action,
            sigma_interval=sigma_end - sigma_start,
            action=action,
            action_is_pad=action_is_pad,
        ), loss_dict

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

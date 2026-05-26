from typing import Optional

import torch
import torch.nn as nn

from .fastwam_one_step_action import FastWAMOneStepAction
from .wan_video_dit import sinusoidal_embedding_1d


class FastWAMOneStepShortcut(FastWAMOneStepAction):
    """Action-only shortcut fine-tuning variant for one-step action generation."""

    @classmethod
    def from_wan22_pretrained(
        cls,
        *,
        one_step_shortcut_start_timestep: Optional[float] = None,
        shortcut_step_size: float = 1.0,
        loss_lambda_action_velocity: float = 0.25,
        loss_lambda_action_endpoint: float = 0.25,
        loss_lambda_shortcut_consistency: float = 0.25,
        loss_lambda_shortcut_half_velocity: float = 0.25,
        freeze_video_expert: bool = True,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(
            one_step_action_timestep=one_step_shortcut_start_timestep,
            loss_lambda_action_velocity=loss_lambda_action_velocity,
            loss_lambda_action_endpoint=loss_lambda_action_endpoint,
            freeze_video_expert=freeze_video_expert,
            **kwargs,
        )
        model.shortcut_step_size = float(shortcut_step_size)
        model.loss_lambda_shortcut_consistency = float(loss_lambda_shortcut_consistency)
        model.loss_lambda_shortcut_half_velocity = float(loss_lambda_shortcut_half_velocity)
        model._install_shortcut_step_conditioner()
        return model

    def _install_shortcut_step_conditioner(self) -> None:
        if hasattr(self.action_expert, "shortcut_step_embedding"):
            return
        hidden_dim = int(self.action_expert.hidden_dim)
        freq_dim = int(self.action_expert.freq_dim)
        self.action_expert.shortcut_step_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_expert.shortcut_step_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )
        final = self.action_expert.shortcut_step_projection[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.action_expert.shortcut_step_embedding.to(device=self.device, dtype=self.torch_dtype)
        self.action_expert.shortcut_step_projection.to(device=self.device, dtype=self.torch_dtype)

    def configure_trainable_parameters(self):
        super().configure_trainable_parameters()
        self._install_shortcut_step_conditioner()
        self.action_expert.shortcut_step_embedding.train()
        self.action_expert.shortcut_step_projection.train()
        self.action_expert.shortcut_step_embedding.requires_grad_(True)
        self.action_expert.shortcut_step_projection.requires_grad_(True)

    def extra_trainable_parameters(self):
        self._install_shortcut_step_conditioner()
        yield from self.action_expert.shortcut_step_embedding.parameters()
        yield from self.action_expert.shortcut_step_projection.parameters()

    def _build_shortcut_step_size(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        step_size = float(self.shortcut_step_size)
        if step_size <= 0.0 or step_size > 1.0:
            raise ValueError(f"`shortcut_step_size` must be in (0, 1], got {step_size}.")
        return torch.full((batch_size,), step_size, device=self.device, dtype=dtype)

    def _step_view(self, step_size: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        return step_size.to(device=sample.device, dtype=sample.dtype).view(
            sample.shape[0],
            *([1] * (sample.ndim - 1)),
        )

    def _apply_shortcut_step_conditioning(
        self,
        action_pre: dict,
        shortcut_step_size: torch.Tensor,
    ) -> dict:
        self._install_shortcut_step_conditioner()
        if shortcut_step_size.ndim != 1:
            raise ValueError(
                f"`shortcut_step_size` must be 1D [B], got shape {tuple(shortcut_step_size.shape)}"
            )
        step_position = shortcut_step_size.to(
            device=action_pre["t_mod"].device,
            dtype=action_pre["t_mod"].dtype,
        ) * float(self.train_action_scheduler.num_train_timesteps)
        step_emb = sinusoidal_embedding_1d(self.action_expert.freq_dim, step_position)
        step_emb = step_emb.to(device=action_pre["t_mod"].device, dtype=action_pre["t_mod"].dtype)
        step_hidden = self.action_expert.shortcut_step_embedding(step_emb)
        step_mod = self.action_expert.shortcut_step_projection(step_hidden).unflatten(
            1,
            (6, self.action_expert.hidden_dim),
        )
        action_pre = dict(action_pre)
        action_pre["t_mod"] = action_pre["t_mod"] + step_mod
        return action_pre

    def _predict_shortcut_action_velocity(
        self,
        *,
        first_frame_latents: torch.Tensor,
        action_tokens: torch.Tensor,
        timestep_action: torch.Tensor,
        shortcut_step_size: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            device=self.device,
            dtype=first_frame_latents.dtype,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self._apply_shortcut_step_conditioning(
            action_pre=action_pre,
            shortcut_step_size=shortcut_step_size,
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
        shortcut_step_size = self._build_shortcut_step_size(
            batch_size=latents_action.shape[0],
            dtype=latents_action.dtype,
        )
        action_pre = self._apply_shortcut_step_conditioning(
            action_pre=action_pre,
            shortcut_step_size=shortcut_step_size,
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

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        first_frame_latents = inputs["first_frame_latents"]
        if first_frame_latents is None:
            first_frame_latents = input_latents[:, :, 0:1]

        noise_action = torch.randn_like(action)
        timestep_start = self._build_one_step_action_timestep(
            batch_size=batch_size,
            dtype=action.dtype,
        )
        shortcut_step_size = self._build_shortcut_step_size(
            batch_size=batch_size,
            dtype=action.dtype,
        )
        max_step_size = timestep_start / float(self.train_action_scheduler.num_train_timesteps)
        if torch.any(shortcut_step_size > max_step_size):
            raise ValueError("`shortcut_step_size` cannot exceed start sigma.")

        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_start)
        target_action_velocity = self.train_action_scheduler.training_target(
            action,
            noise_action,
            timestep_start,
        )

        pred_large_velocity = self._predict_shortcut_action_velocity(
            first_frame_latents=first_frame_latents,
            action_tokens=noisy_action,
            timestep_action=timestep_start,
            shortcut_step_size=shortcut_step_size,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )

        half_step_size = shortcut_step_size * 0.5
        pred_half_1_velocity = self._predict_shortcut_action_velocity(
            first_frame_latents=first_frame_latents,
            action_tokens=noisy_action,
            timestep_action=timestep_start,
            shortcut_step_size=half_step_size,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        mid_action = noisy_action - self._step_view(half_step_size, noisy_action) * pred_half_1_velocity.detach()
        timestep_mid = timestep_start - half_step_size * float(self.train_action_scheduler.num_train_timesteps)
        pred_half_2_velocity = self._predict_shortcut_action_velocity(
            first_frame_latents=first_frame_latents,
            action_tokens=mid_action,
            timestep_action=timestep_mid,
            shortcut_step_size=half_step_size,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )

        large_step = self._step_view(shortcut_step_size, noisy_action)
        pred_large_endpoint = noisy_action - large_step * pred_large_velocity
        target_large_endpoint = noisy_action - large_step * target_action_velocity
        shortcut_target_velocity = 0.5 * (
            pred_half_1_velocity.detach() + pred_half_2_velocity.detach()
        )

        loss_action_velocity = self._masked_action_mse(
            pred=pred_large_velocity,
            target=target_action_velocity,
            action_is_pad=action_is_pad,
        )
        loss_action_endpoint = self._masked_action_mse(
            pred=pred_large_endpoint,
            target=target_large_endpoint,
            action_is_pad=action_is_pad,
        )
        loss_shortcut_consistency = self._masked_action_mse(
            pred=pred_large_velocity,
            target=shortcut_target_velocity,
            action_is_pad=action_is_pad,
        )
        loss_shortcut_half_velocity = 0.5 * (
            self._masked_action_mse(
                pred=pred_half_1_velocity,
                target=target_action_velocity,
                action_is_pad=action_is_pad,
            )
            + self._masked_action_mse(
                pred=pred_half_2_velocity,
                target=target_action_velocity,
                action_is_pad=action_is_pad,
            )
        )

        loss_total = (
            self.loss_lambda_action_velocity * loss_action_velocity
            + self.loss_lambda_action_endpoint * loss_action_endpoint
            + self.loss_lambda_shortcut_consistency * loss_shortcut_consistency
            + self.loss_lambda_shortcut_half_velocity * loss_shortcut_half_velocity
        )
        loss_dict = {
            "loss_action_velocity": self.loss_lambda_action_velocity
            * float(loss_action_velocity.detach().item()),
            "loss_action_endpoint": self.loss_lambda_action_endpoint
            * float(loss_action_endpoint.detach().item()),
            "loss_shortcut_consistency": self.loss_lambda_shortcut_consistency
            * float(loss_shortcut_consistency.detach().item()),
            "loss_shortcut_half_velocity": self.loss_lambda_shortcut_half_velocity
            * float(loss_shortcut_half_velocity.detach().item()),
        }
        return loss_total, loss_dict

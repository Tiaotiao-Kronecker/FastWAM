import math
from typing import Optional

import torch

from .fastwam_one_step_shortcut import FastWAMOneStepShortcut
from .wan_video_dit import sinusoidal_embedding_1d


class FastWAMOneStepShortcutOfficial(FastWAMOneStepShortcut):
    """Shortcut fine-tuning variant closer to the paper's multi-step bootstrap target.

    The FastWAM action scheduler uses sigma=1 for noise and sigma=0 for data, so
    a shortcut step of length d moves from sigma to sigma-d.
    """

    @classmethod
    def from_wan22_pretrained(
        cls,
        *,
        shortcut_num_denoise_steps: int = 128,
        shortcut_bootstrap_ratio: float = 0.75,
        loss_lambda_shortcut_target: float = 1.0,
        one_step_shortcut_start_timestep: Optional[float] = None,
        shortcut_step_size: float = 1.0,
        freeze_video_expert: bool = True,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(
            one_step_shortcut_start_timestep=one_step_shortcut_start_timestep,
            shortcut_step_size=shortcut_step_size,
            freeze_video_expert=freeze_video_expert,
            **kwargs,
        )
        model.shortcut_num_denoise_steps = int(shortcut_num_denoise_steps)
        model.shortcut_bootstrap_ratio = float(shortcut_bootstrap_ratio)
        model.loss_lambda_shortcut_target = float(loss_lambda_shortcut_target)
        model._validate_official_shortcut_config()
        return model

    def _validate_official_shortcut_config(self) -> None:
        steps = int(self.shortcut_num_denoise_steps)
        if steps <= 1 or steps & (steps - 1) != 0:
            raise ValueError(
                "`shortcut_num_denoise_steps` must be a power of two greater than 1, "
                f"got {steps}."
            )
        ratio = float(self.shortcut_bootstrap_ratio)
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"`shortcut_bootstrap_ratio` must be in [0, 1], got {ratio}.")

    @property
    def _max_shortcut_dt_base(self) -> int:
        return int(math.log2(int(self.shortcut_num_denoise_steps)))

    def _apply_shortcut_dt_base_conditioning(
        self,
        action_pre: dict,
        shortcut_dt_base: torch.Tensor,
    ) -> dict:
        self._install_shortcut_step_conditioner()
        if shortcut_dt_base.ndim != 1:
            raise ValueError(
                f"`shortcut_dt_base` must be 1D [B], got shape {tuple(shortcut_dt_base.shape)}"
            )
        dt_position = shortcut_dt_base.to(
            device=action_pre["t_mod"].device,
            dtype=action_pre["t_mod"].dtype,
        )
        dt_emb = sinusoidal_embedding_1d(self.action_expert.freq_dim, dt_position)
        dt_emb = dt_emb.to(device=action_pre["t_mod"].device, dtype=action_pre["t_mod"].dtype)
        dt_hidden = self.action_expert.shortcut_step_embedding(dt_emb)
        dt_mod = self.action_expert.shortcut_step_projection(dt_hidden).unflatten(
            1,
            (6, self.action_expert.hidden_dim),
        )
        action_pre = dict(action_pre)
        action_pre["t_mod"] = action_pre["t_mod"] + dt_mod
        return action_pre

    def _apply_shortcut_step_conditioning(
        self,
        action_pre: dict,
        shortcut_step_size: torch.Tensor,
    ) -> dict:
        step_size = shortcut_step_size.to(device=action_pre["t_mod"].device, dtype=torch.float32)
        if torch.any(step_size <= 0.0) or torch.any(step_size > 1.0):
            raise ValueError("`shortcut_step_size` must be in (0, 1].")
        shortcut_dt_base = -torch.log2(step_size).round()
        return self._apply_shortcut_dt_base_conditioning(
            action_pre=action_pre,
            shortcut_dt_base=shortcut_dt_base.to(dtype=action_pre["t_mod"].dtype),
        )

    def _predict_shortcut_dt_action_velocity(
        self,
        *,
        video_pre: dict,
        action_tokens: torch.Tensor,
        timestep_action: torch.Tensor,
        shortcut_dt_base: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self._apply_shortcut_dt_base_conditioning(
            action_pre=action_pre,
            shortcut_dt_base=shortcut_dt_base,
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

    def _sample_flow_shortcut_batch(
        self,
        *,
        action: torch.Tensor,
        noise_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = action.shape[0]
        num_sections = int(self.shortcut_num_denoise_steps)
        section_index = torch.randint(
            low=0,
            high=num_sections,
            size=(batch_size,),
            device=action.device,
        )
        sigma = 1.0 - section_index.to(dtype=action.dtype) / float(num_sections)
        timestep_action = sigma * float(self.train_action_scheduler.num_train_timesteps)
        shortcut_dt_base = torch.full(
            (batch_size,),
            float(self._max_shortcut_dt_base),
            device=action.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action_velocity = self.train_action_scheduler.training_target(
            action,
            noise_action,
            timestep_action,
        )
        return noisy_action, timestep_action, shortcut_dt_base, target_action_velocity

    def _sample_bootstrap_shortcut_batch(
        self,
        *,
        action: torch.Tensor,
        noise_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = action.shape[0]
        max_dt_base = self._max_shortcut_dt_base
        dt_base_int = torch.randint(
            low=0,
            high=max_dt_base,
            size=(batch_size,),
            device=action.device,
        )
        shortcut_dt_base = dt_base_int.to(dtype=action.dtype)
        num_sections = torch.pow(
            torch.full((batch_size,), 2.0, device=action.device, dtype=action.dtype),
            shortcut_dt_base,
        )
        section_index = torch.floor(torch.rand((batch_size,), device=action.device, dtype=action.dtype) * num_sections)
        sigma_start = 1.0 - section_index / num_sections
        step_size = 1.0 / num_sections
        timestep_action = sigma_start * float(self.train_action_scheduler.num_train_timesteps)
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        return noisy_action, timestep_action, shortcut_dt_base, step_size, shortcut_dt_base + 1.0

    def _video_pre(
        self,
        *,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> dict:
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            device=self.device,
            dtype=first_frame_latents.dtype,
        )
        return self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )

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

        video_pre = self._video_pre(
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        noise_action = torch.randn_like(action)
        use_bootstrap = (
            self._max_shortcut_dt_base > 0
            and torch.rand((), device=action.device).item() < float(self.shortcut_bootstrap_ratio)
        )

        if use_bootstrap:
            noisy_action, timestep_action, shortcut_dt_base, step_size, half_dt_base = (
                self._sample_bootstrap_shortcut_batch(
                    action=action,
                    noise_action=noise_action,
                )
            )
            half_step = step_size * 0.5
            with torch.no_grad():
                pred_half_1_velocity = self._predict_shortcut_dt_action_velocity(
                    video_pre=video_pre,
                    action_tokens=noisy_action,
                    timestep_action=timestep_action,
                    shortcut_dt_base=half_dt_base,
                    context=context,
                    context_mask=context_mask,
                )
                half_step_view = half_step.view(batch_size, *([1] * (action.ndim - 1)))
                mid_action = noisy_action - half_step_view * pred_half_1_velocity
                timestep_mid = timestep_action - half_step * float(self.train_action_scheduler.num_train_timesteps)
                pred_half_2_velocity = self._predict_shortcut_dt_action_velocity(
                    video_pre=video_pre,
                    action_tokens=mid_action,
                    timestep_action=timestep_mid,
                    shortcut_dt_base=half_dt_base,
                    context=context,
                    context_mask=context_mask,
                )
                target_velocity = 0.5 * (pred_half_1_velocity + pred_half_2_velocity)
            target_kind = "bootstrap"
        else:
            noisy_action, timestep_action, shortcut_dt_base, target_velocity = self._sample_flow_shortcut_batch(
                action=action,
                noise_action=noise_action,
            )
            target_kind = "flow"

        pred_velocity = self._predict_shortcut_dt_action_velocity(
            video_pre=video_pre,
            action_tokens=noisy_action,
            timestep_action=timestep_action,
            shortcut_dt_base=shortcut_dt_base,
            context=context,
            context_mask=context_mask,
        )
        loss_shortcut_target = self._masked_action_mse(
            pred=pred_velocity,
            target=target_velocity,
            action_is_pad=action_is_pad,
        )
        loss_total = self.loss_lambda_shortcut_target * loss_shortcut_target
        loss_value = self.loss_lambda_shortcut_target * float(loss_shortcut_target.detach().item())
        loss_dict = {
            "loss_shortcut_target": loss_value,
            "loss_shortcut_flow": loss_value if target_kind == "flow" else 0.0,
            "loss_shortcut_bootstrap": loss_value if target_kind == "bootstrap" else 0.0,
        }
        return loss_total, loss_dict

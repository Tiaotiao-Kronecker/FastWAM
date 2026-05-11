from typing import Optional

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM

logger = get_logger(__name__)


class FastWAMOneStepAction(FastWAM):
    """Action-only one-step endpoint fine-tuning variant."""

    @classmethod
    def from_wan22_pretrained(
        cls,
        *,
        one_step_action_timestep: Optional[float] = None,
        loss_lambda_action_velocity: float = 0.5,
        loss_lambda_action_endpoint: float = 0.5,
        freeze_video_expert: bool = True,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(**kwargs)
        model.one_step_action_timestep = one_step_action_timestep
        model.loss_lambda_action_velocity = float(loss_lambda_action_velocity)
        model.loss_lambda_action_endpoint = float(loss_lambda_action_endpoint)
        model.freeze_video_expert = bool(freeze_video_expert)
        return model

    def configure_trainable_parameters(self):
        if getattr(self, "freeze_video_expert", True):
            self.video_expert.eval()
            self.video_expert.requires_grad_(False)
            logger.info("FastWAMOneStepAction: frozen video expert parameters.")

    def _build_one_step_action_timestep(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        timestep = self.one_step_action_timestep
        if timestep is None:
            timestep = float(self.train_action_scheduler.num_train_timesteps)
        timestep = float(timestep)
        if timestep <= 0 or timestep > float(self.train_action_scheduler.num_train_timesteps):
            raise ValueError(
                "`one_step_action_timestep` must be in (0, action_num_train_timesteps], "
                f"got {timestep}."
            )
        return torch.full((batch_size,), timestep, device=self.device, dtype=dtype)

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

        timestep_video = torch.zeros(
            (batch_size,),
            device=self.device,
            dtype=input_latents.dtype,
        )

        noise_action = torch.randn_like(action)
        timestep_action = self._build_one_step_action_timestep(
            batch_size=batch_size,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action_velocity = self.train_action_scheduler.training_target(
            action,
            noise_action,
            timestep_action,
        )

        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
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

        pred_action_velocity = self.action_expert.post_dit(tokens_out["action"], action_pre)
        sigma = (timestep_action / float(self.train_action_scheduler.num_train_timesteps)).to(
            device=noisy_action.device,
            dtype=noisy_action.dtype,
        )
        sigma = sigma.view(batch_size, *([1] * (noisy_action.ndim - 1)))
        pred_action_endpoint = noisy_action - sigma * pred_action_velocity

        loss_action_velocity = self._masked_action_mse(
            pred=pred_action_velocity,
            target=target_action_velocity,
            action_is_pad=action_is_pad,
        )
        loss_action_endpoint = self._masked_action_mse(
            pred=pred_action_endpoint,
            target=action,
            action_is_pad=action_is_pad,
        )

        loss_total = (
            self.loss_lambda_action_velocity * loss_action_velocity
            + self.loss_lambda_action_endpoint * loss_action_endpoint
        )
        loss_dict = {
            "loss_action_velocity": self.loss_lambda_action_velocity
            * float(loss_action_velocity.detach().item()),
            "loss_action_endpoint": self.loss_lambda_action_endpoint
            * float(loss_action_endpoint.detach().item()),
        }
        return loss_total, loss_dict

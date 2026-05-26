from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fastwam_joint import FastWAMJoint
from .wan_video_dit import sinusoidal_embedding_1d


class FastWAMJointMeanFlow(FastWAMJoint):
    """Joint video/action MeanFlow fine-tuning variant with finite-difference training."""

    @classmethod
    def from_wan22_pretrained(
        cls,
        *,
        meanflow_start_timestep: Optional[float] = 0.0,
        meanflow_end_timestep: Optional[float] = None,
        meanflow_derivative_epsilon: float = 0.05,
        meanflow_random_timesteps: bool = True,
        meanflow_equal_time_prob: float = 0.0,
        meanflow_trainable_scope: str = "joint",
        meanflow_train_proprio_encoder: bool = True,
        loss_lambda_meanflow_video: float = 1.0,
        loss_lambda_meanflow_action: float = 1.0,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(**kwargs)
        model.meanflow_start_timestep = meanflow_start_timestep
        model.meanflow_end_timestep = meanflow_end_timestep
        model.meanflow_derivative_epsilon = float(meanflow_derivative_epsilon)
        model.meanflow_random_timesteps = bool(meanflow_random_timesteps)
        model.meanflow_equal_time_prob = float(meanflow_equal_time_prob)
        model.meanflow_trainable_scope = str(meanflow_trainable_scope).strip().lower()
        model.meanflow_train_proprio_encoder = bool(meanflow_train_proprio_encoder)
        model.loss_lambda_meanflow_video = float(loss_lambda_meanflow_video)
        model.loss_lambda_meanflow_action = float(loss_lambda_meanflow_action)
        model._install_meanflow_conditioners()
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

    def _install_meanflow_conditioners(self) -> None:
        if hasattr(self.video_expert, "meanflow_start_embedding") and hasattr(self.action_expert, "meanflow_start_embedding"):
            return

        def _make_embedding(freq_dim: int, hidden_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(freq_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def _make_projection(hidden_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim * 6),
            )

        video_hidden_dim = int(self.video_expert.hidden_dim)
        video_freq_dim = int(self.video_expert.freq_dim)
        action_hidden_dim = int(self.action_expert.hidden_dim)
        action_freq_dim = int(self.action_expert.freq_dim)

        self.video_expert.meanflow_start_embedding = _make_embedding(video_freq_dim, video_hidden_dim).to(
            device=self.device, dtype=self.torch_dtype
        )
        self.video_expert.meanflow_start_projection = _make_projection(video_hidden_dim).to(
            device=self.device, dtype=self.torch_dtype
        )
        self.action_expert.meanflow_start_embedding = _make_embedding(action_freq_dim, action_hidden_dim).to(
            device=self.device, dtype=self.torch_dtype
        )
        self.action_expert.meanflow_start_projection = _make_projection(action_hidden_dim).to(
            device=self.device, dtype=self.torch_dtype
        )

        nn.init.zeros_(self.video_expert.meanflow_start_projection[-1].weight)
        nn.init.zeros_(self.video_expert.meanflow_start_projection[-1].bias)
        nn.init.zeros_(self.action_expert.meanflow_start_projection[-1].weight)
        nn.init.zeros_(self.action_expert.meanflow_start_projection[-1].bias)

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
        if not getattr(self, "meanflow_random_timesteps", True):
            timestep_start = self._build_meanflow_start_timestep(batch_size=batch_size, dtype=dtype)
            timestep_end = self._build_meanflow_end_timestep(batch_size=batch_size, dtype=dtype)
            steps = float(self.train_action_scheduler.num_train_timesteps)
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

    def _apply_meanflow_conditioning(
        self,
        pre_state: dict,
        timestep_start: torch.Tensor,
        *,
        branch: str,
    ) -> dict:
        self._install_meanflow_conditioners()
        if branch == "video":
            embedding = self.video_expert.meanflow_start_embedding
            projection = self.video_expert.meanflow_start_projection
        elif branch == "action":
            embedding = self.action_expert.meanflow_start_embedding
            projection = self.action_expert.meanflow_start_projection
        else:
            raise ValueError(f"Unknown branch: {branch}")

        start_position = timestep_start.to(device=pre_state["t_mod"].device, dtype=pre_state["t_mod"].dtype)
        start_emb = sinusoidal_embedding_1d(embedding[0].in_features, start_position)
        start_emb = start_emb.to(device=pre_state["t_mod"].device, dtype=pre_state["t_mod"].dtype)
        start_hidden = embedding(start_emb)
        start_mod = projection(start_hidden).unflatten(1, (6, start_hidden.shape[-1]))
        if pre_state["t_mod"].ndim == 4:
            start_mod = start_mod.unsqueeze(1)
        elif pre_state["t_mod"].ndim != 3:
            raise ValueError(f"Unsupported `t_mod` shape: {tuple(pre_state['t_mod'].shape)}")
        pre_state = dict(pre_state)
        pre_state["t_mod"] = pre_state["t_mod"] + start_mod
        return pre_state

    def _predict_joint_mean_velocity(
        self,
        *,
        video_pre: dict,
        action_pre: dict,
        video_timestep_start: torch.Tensor,
        action_timestep_start: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self._apply_meanflow_conditioning(video_pre, video_timestep_start, branch="video")
        action_pre = self._apply_meanflow_conditioning(action_pre, action_timestep_start, branch="action")
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
        action_pre = self._apply_meanflow_conditioning(
            action_pre=action_pre,
            timestep_start=timestep_start,
            branch="action",
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
        batch_size = latents_action.shape[0]
        sigma_start = self._build_meanflow_start_timestep(batch_size=batch_size, dtype=torch.float32)
        sigma_start = sigma_start / float(self.train_action_scheduler.num_train_timesteps)
        timestep_video_start = (sigma_start * float(self.train_video_scheduler.num_train_timesteps)).to(
            device=self.device,
            dtype=timestep_video.dtype,
        )
        timestep_action_start = (sigma_start * float(self.train_action_scheduler.num_train_timesteps)).to(
            device=self.device,
            dtype=timestep_action.dtype,
        )

        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        return self._predict_joint_mean_velocity(
            video_pre=video_pre,
            action_pre=action_pre,
            video_timestep_start=timestep_video_start,
            action_timestep_start=timestep_action_start,
        )

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
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        sigma_start = self._build_meanflow_start_timestep(
            batch_size=first_frame_latents.shape[0],
            dtype=torch.float32,
        ) / float(self.train_action_scheduler.num_train_timesteps)
        video_timestep_start = (
            sigma_start * float(self.train_video_scheduler.num_train_timesteps)
        ).to(device=self.device, dtype=first_frame_latents.dtype)
        video_pre = self._apply_meanflow_conditioning(
            pre_state=video_pre,
            timestep_start=video_timestep_start,
            branch="video",
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(pred_action_posi, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    def configure_trainable_parameters(self):
        super().configure_trainable_parameters()
        self._install_meanflow_conditioners()

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
            self._set_meanflow_conditioners_trainable(True)
        elif scope == "conditioner":
            self.video_expert.eval()
            self.video_expert.requires_grad_(False)
            self.action_expert.eval()
            self.action_expert.requires_grad_(False)
            self._set_meanflow_conditioners_trainable(True)
        else:
            raise ValueError(
                "`meanflow_trainable_scope` must be one of ['joint', 'conditioner'], "
                f"got {scope!r}."
            )

    def _set_meanflow_conditioners_trainable(self, trainable: bool) -> None:
        self._install_meanflow_conditioners()
        modules = (
            self.video_expert.meanflow_start_embedding,
            self.video_expert.meanflow_start_projection,
            self.action_expert.meanflow_start_embedding,
            self.action_expert.meanflow_start_projection,
        )
        for module in modules:
            module.train(trainable)
            module.requires_grad_(trainable)

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        video_noise = torch.randn_like(input_latents)
        action_noise = torch.randn_like(action)
        sigma_start, sigma_end = self._sample_meanflow_sigma_pair(
            batch_size=batch_size,
            dtype=torch.float32,
            equal_time_prob=0.0,
            min_interval=float(self.meanflow_derivative_epsilon),
        )
        sigma_interval = sigma_end - sigma_start
        prev_sigma = torch.maximum(sigma_start, sigma_end - float(self.meanflow_derivative_epsilon))
        prev_sigma_interval = sigma_end - prev_sigma

        timestep_start_video = sigma_start * float(self.train_video_scheduler.num_train_timesteps)
        timestep_end_video = sigma_end * float(self.train_video_scheduler.num_train_timesteps)
        timestep_start_action = sigma_start * float(self.train_action_scheduler.num_train_timesteps)
        timestep_end_action = sigma_end * float(self.train_action_scheduler.num_train_timesteps)
        prev_timestep_video = prev_sigma * float(self.train_video_scheduler.num_train_timesteps)
        prev_timestep_action = prev_sigma * float(self.train_action_scheduler.num_train_timesteps)

        noisy_video = self.train_video_scheduler.add_noise(input_latents, video_noise, timestep_end_video)
        noisy_action = self.train_action_scheduler.add_noise(action, action_noise, timestep_end_action)
        if inputs["first_frame_latents"] is not None:
            noisy_video[:, :, 0:1] = inputs["first_frame_latents"]
        target_video_velocity = self.train_video_scheduler.training_target(input_latents, video_noise, timestep_end_video)
        target_action_velocity = self.train_action_scheduler.training_target(action, action_noise, timestep_end_action)

        video_pre = self.video_expert.pre_dit(
            x=noisy_video,
            timestep=timestep_end_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_end_action,
            context=context,
            context_mask=context_mask,
        )

        pred_video_velocity, pred_action_velocity = self._predict_joint_mean_velocity(
            video_pre=video_pre,
            action_pre=action_pre,
            video_timestep_start=timestep_start_video,
            action_timestep_start=timestep_start_action,
        )

        eps = float(self.meanflow_derivative_epsilon)
        if eps <= 0.0 or eps >= 1.0:
            raise ValueError(f"`meanflow_derivative_epsilon` must be in (0, 1), got {eps}.")
        if torch.any(prev_sigma >= sigma_end):
            raise ValueError("Mean-flow finite-difference epsilon collapsed to zero.")

        prev_video = noisy_video - self._sigma_view(prev_sigma_interval, input_latents) * target_video_velocity
        prev_action = noisy_action - self._sigma_view(prev_sigma_interval, action) * target_action_velocity
        if inputs["first_frame_latents"] is not None:
            prev_video[:, :, 0:1] = inputs["first_frame_latents"]

        with torch.no_grad():
            prev_video_pre = self.video_expert.pre_dit(
                x=prev_video,
                timestep=prev_timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )
            prev_action_pre = self.action_expert.pre_dit(
                action_tokens=prev_action,
                timestep=prev_timestep_action,
                context=context,
                context_mask=context_mask,
            )
            pred_prev_video_velocity, pred_prev_action_velocity = self._predict_joint_mean_velocity(
                video_pre=prev_video_pre,
                action_pre=prev_action_pre,
                video_timestep_start=timestep_start_video,
                action_timestep_start=timestep_start_action,
            )
            dudt_video = (pred_video_velocity.detach() - pred_prev_video_velocity) / self._sigma_view(
                prev_sigma_interval,
                input_latents,
            )
            dudt_action = (pred_action_velocity.detach() - pred_prev_action_velocity) / self._sigma_view(
                prev_sigma_interval,
                action,
            )
            meanflow_target_video = target_video_velocity - self._sigma_view(sigma_interval, input_latents) * dudt_video
            meanflow_target_action = target_action_velocity - self._sigma_view(sigma_interval, action) * dudt_action

        pred_video_endpoint = noisy_video - self._sigma_view(sigma_interval, input_latents) * pred_video_velocity
        pred_action_endpoint = noisy_action - self._sigma_view(sigma_interval, action) * pred_action_velocity

        include_initial_video_step = inputs["first_frame_latents"] is None
        loss_pred_video_velocity = pred_video_velocity
        loss_meanflow_target_video = meanflow_target_video
        loss_pred_video_endpoint = pred_video_endpoint
        loss_input_latents = input_latents
        if not include_initial_video_step:
            loss_pred_video_velocity = loss_pred_video_velocity[:, :, 1:]
            loss_meanflow_target_video = loss_meanflow_target_video[:, :, 1:]
            loss_pred_video_endpoint = loss_pred_video_endpoint[:, :, 1:]
            loss_input_latents = loss_input_latents[:, :, 1:]

        loss_video = self._compute_video_loss_per_sample(
            pred_video=loss_pred_video_velocity,
            target_video=loss_meanflow_target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        ).mean()
        loss_action = self._masked_action_mse(
            pred=pred_action_velocity,
            target=meanflow_target_action,
            action_is_pad=action_is_pad,
        )

        loss_total = self.loss_lambda_meanflow_video * loss_video + self.loss_lambda_meanflow_action * loss_action
        loss_dict = {
            "loss_meanflow_video": self.loss_lambda_meanflow_video * float(loss_video.detach().item()),
            "loss_meanflow_action": self.loss_lambda_meanflow_action * float(loss_action.detach().item()),
            "meanflow_sigma_start": float(sigma_start.detach().float().mean().item()),
            "meanflow_sigma_end": float(sigma_end.detach().float().mean().item()),
            "meanflow_interval": float(sigma_interval.detach().float().mean().item()),
        }
        if self.loss_lambda_meanflow_video != 0.0:
            loss_dict["loss_video_endpoint_sanity"] = float(
                self._compute_video_loss_per_sample(
                    pred_video=loss_pred_video_endpoint,
                    target_video=loss_input_latents,
                    image_is_pad=image_is_pad,
                    include_initial_video_step=include_initial_video_step,
                ).mean().detach().item()
            )
        if self.loss_lambda_meanflow_action != 0.0:
            loss_dict["loss_action_endpoint_sanity"] = float(
                self._masked_action_mse(
                    pred=pred_action_endpoint,
                    target=action,
                    action_is_pad=action_is_pad,
                ).detach().item()
            )
        return loss_total, loss_dict

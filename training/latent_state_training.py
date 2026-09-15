"""Two-stage objectives and mode configuration for annotation-free training."""
import os

import torch
from torch.nn import functional as F


LATENT_MODEL_DEFAULTS = {
    "latent_state_num_tokens": 4, "latent_state_dim": 256,
    "latent_state_initializer_layers": 2, "latent_state_transition_layers": 2,
    "latent_state_num_heads": 8, "latent_state_mlp_ratio": 4.0,
    "latent_state_dropout": 0.0, "latent_state_max_frames": 64,
    "latent_state_bidirectional": False,
    "transition_tokenizer_num_tokens": 4, "transition_tokenizer_dim": 256,
    "transition_tokenizer_layers": 2, "transition_tokenizer_heads": 8,
    "transition_tokenizer_mlp_ratio": 4.0, "transition_tokenizer_dropout": 0.0,
    "transition_tokenizer_fsq_levels": (8, 8, 8), "transition_vocab_size": 512,
}
TRAIN_MODES = ("transition_tokenizer", "latent_state_world_model",
               "explicit_state_baseline", "latent_state_joint_finetune")


def resolve_mode_config(train_mode, **overrides):
    if train_mode == "latent_state_joint_finetune":
        raise NotImplementedError("Joint tokenizer fine-tuning is reserved; use the two-stage modes")
    if train_mode not in TRAIN_MODES:
        raise ValueError(f"Unknown train_mode: {train_mode}")
    baseline = train_mode == "explicit_state_baseline"
    stage1 = train_mode == "transition_tokenizer"
    required = {"use_latent_state": not baseline and not stage1,
                "use_explicit_state": baseline, "use_transition_tokenizer": not baseline,
                "freeze_transition_tokenizer": not baseline and not stage1}
    for key, value in required.items():
        if overrides.get(key) is not None and overrides[key] != value:
            raise ValueError(f"{train_mode} requires {key}={value}")
    return required


def load_transition_tokenizer(tokenizer, path):
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if "transition_tokenizer" in state and isinstance(state["transition_tokenizer"], dict):
        state = state["transition_tokenizer"]
    for prefix in ("module.pipe.dit.transition_tokenizer.", "pipe.dit.transition_tokenizer.",
                   "dit.transition_tokenizer.", "transition_tokenizer."):
        selected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if selected:
            state = selected
            break
    if "quantizer.levels" in state and not torch.equal(
            state["quantizer.levels"].cpu(), tokenizer.quantizer.levels.cpu()):
        raise ValueError("Tokenizer checkpoint FSQ levels disagree with the configured quantizer")
    tokenizer.load_state_dict(state, strict=True)


def _log_components(pipe, mode, total, video=None, transition=None, tokenizer=None):
    values = {"video_loss": video, "transition_loss": transition, "tokenizer_loss": tokenizer}
    pipe.last_loss_components = {key: 0.0 if value is None else value.detach().item()
                                 for key, value in values.items()}
    pipe.last_loss_components.update(state_loss=0.0, total_loss=total.detach().item())
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[loss_components] mode={mode} " + " ".join(
            f"{key}={value:.6f}" for key, value in pipe.last_loss_components.items()), flush=True)


def _log_shapes(pipe, mode, inputs, shape, clean_shape, z_indices, outputs=None):
    if getattr(pipe, "_latent_shapes_logged", False):
        return
    if int(os.environ.get("RANK", "0")) == 0:
        def size(value):
            return None if value is None else tuple(value.shape)
        fields = {"input_latents": size(inputs["input_latents"]),
                  "latents": size(inputs.get("latents")), "patchified": shape["patchified"],
                  "clean_visual_tokens": clean_shape,
                  "first_frame_tokens": (clean_shape[0], shape["tokens_per_frame"], clean_shape[-1]),
                  "f": shape["f"], "tokens_per_frame": shape["tokens_per_frame"],
                  "z_indices": size(z_indices)}
        for key in ("action_binned", "G_seq_low", "G_seq", "transition_logits"):
            fields[key] = size((outputs or {}).get(key))
        print(f"[LatentState shapes] train_mode={mode} " + " ".join(
            f"{key}={value}" for key, value in fields.items()), flush=True)
    pipe._latent_shapes_logged = True


def TransitionTokenizerLoss(pipe, tokenizer_loss_weight=1.0, **inputs):
    with torch.no_grad():
        clean_tokens, shape = pipe.dit.extract_clean_visual_tokens(inputs["input_latents"])
    result = pipe.dit.transition_tokenizer(clean_tokens, latent_frames=shape["f"])
    total = float(tokenizer_loss_weight) * result["loss_tok"]
    _log_shapes(pipe, "transition_tokenizer", inputs, shape, tuple(clean_tokens.shape), result["z_indices"])
    _log_components(pipe, "transition_tokenizer", total, tokenizer=result["loss_tok"])
    return total


def FlowMatchSFTLossWithLatentState(pipe, video_loss_weight=1.0, transition_loss_weight=0.1,
                                   use_transition_loss=True, **inputs):
    # Labels never enter the latent forward, including when supplied by a cached baseline dataset.
    inputs.pop("state", None)
    inputs.pop("noisy_state", None)
    clean = inputs["input_latents"]
    if clean.shape[2] < 2:
        raise ValueError("Transition supervision needs at least two clean latent frames")
    lower = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    upper = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(lower, upper, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(device=pipe.device, dtype=pipe.torch_dtype)
    noise = torch.randn_like(clean)
    inputs["latents"] = pipe.scheduler.add_noise(clean, noise, timestep)
    target = pipe.scheduler.training_target(clean, noise, timestep)
    # A clean initial observation is required even without --extra_inputs input_image.
    first = inputs.get("first_frame_latents", clean[:, :, :1])
    inputs["first_frame_latents"] = first
    inputs["latents"][:, :, :1] = first
    inputs["fuse_vae_embedding_in_latents"] = True
    pipe.dit.transition_tokenizer.eval()
    with torch.no_grad():
        clean_tokens, shape = pipe.dit.extract_clean_visual_tokens(clean)
        clean_shape = tuple(clean_tokens.shape)
        result = pipe.dit.transition_tokenizer(clean_tokens, latent_frames=shape["f"])
        z_indices = result["z_indices"].detach()
    del result, clean_tokens
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    outputs = pipe.model_fn(**models, **inputs, timestep=timestep)
    pred = outputs["pred_video"][:, :, 1:]
    loss_video = F.mse_loss(pred.float(), target[:, :, 1:].float()) * pipe.scheduler.training_weight(timestep)
    logits = outputs["transition_logits"]
    if logits.shape[:-1] != z_indices.shape:
        raise ValueError(f"Transition shapes differ: {logits.shape} vs {z_indices.shape}")
    loss_trans = F.cross_entropy(logits.flatten(0, 2).float(), z_indices.flatten())
    total = float(video_loss_weight) * loss_video
    total = total + (float(transition_loss_weight) if use_transition_loss else 0.0) * loss_trans
    _log_shapes(pipe, "latent_state_world_model", inputs, shape, clean_shape, z_indices, outputs)
    _log_components(pipe, "latent_state_world_model", total, video=loss_video, transition=loss_trans)
    return total

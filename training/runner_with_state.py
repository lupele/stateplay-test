"""Training launcher with seamless resume support (StatePlay-local).

Mirrors `diffsynth.diffusion.runner.launch_training_task` byte-for-byte,
adding TWO things and ONLY two things:

  (1) After `accelerator.prepare(...)`, if `args.resume_state` is set,
      call `accelerator.load_state(args.resume_state)` — restores optimizer
      moments, LR scheduler state, RNG, and DataLoader sampler state
      (dataloader state restored only for prepared DataLoaders w/ workers
      below accelerate's checkpointable threshold).

  (2) At each save_steps boundary, if `args.save_full_state` is True, call
      `accelerator.save_state(<output>/state-<N>/)` ALONGSIDE the existing
      `step-<N>.safetensors` written by `ModelLogger.on_step_end`. The two
      products co-exist; `step-<N>.safetensors` remains the canonical
      lightweight (trainable-only) checkpoint, `state-<N>/` is the full
      Accelerate state directory used by `--resume_state`.

No upstream `diffsynth/` code is touched. `train.py` dispatches into this
function via `launcher_map["sft"] = launch_training_task_with_state`.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm

from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from diffsynth.diffusion.training_module import DiffusionTrainingModule


def collate_samples(samples):
    return samples


def launch_training_task_with_state(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: Optional[int] = None,
    num_epochs: int = 1,
    args=None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    max_train_steps = None if args is None else getattr(args, "max_train_steps", None)
    save_full_state = bool(args is not None and getattr(args, "save_full_state", False))
    resume_state_path = (args.resume_state if args is not None else None)
    smoke = bool(args is not None and getattr(args, "smoke_test", False))

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay, foreach=False
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=getattr(args, "batch_size", 4), shuffle=True,
        collate_fn=collate_samples, num_workers=num_workers,
    )
    model.to(device=accelerator.device)
    if getattr(model, "offload_frozen_encoders", False):
        frozen_prefixes = tuple(
            "pipe." + name + "." for name in ("text_encoder", "vae")
            if getattr(model.pipe, name, None) is not None
            and not any(p.requires_grad for p in getattr(model.pipe, name).parameters())
        )
        model._ddp_params_and_buffers_to_ignore = [
            name for name, _ in list(model.named_parameters()) + list(model.named_buffers())
            if name.startswith(frozen_prefixes)
        ]
    original_model = model
    offload_encoders = getattr(model, "offload_frozen_encoders", False)
    if accelerator.distributed_type.name == "DEEPSPEED":
        # DeepSpeed broadcasts all parameters once; frozen encoders must be on CUDA for that broadcast.
        model.offload_frozen_encoders = False
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    original_model.offload_frozen_encoders = offload_encoders
    original_model.pipe.load_models_to_device(("dit",))
    smoke_gradients = {}
    smoke_hooks = []
    if smoke:
        names = ("latent_state_rollout.initializer.queries", "latent_state_rollout.transition.delta.weight",
                 "latent_transition_predictor.mlp.3.weight", "blocks.0.latent_state_attention.gate",
                 "transition_tokenizer.queries")
        for name, parameter in original_model.pipe.dit.named_parameters():
            if name in names and parameter.requires_grad:
                def record(gradient, key=name):
                    smoke_gradients[key] = gradient.detach().float().norm()
                smoke_hooks.append(parameter.register_hook(record))
    initialize_deepspeed_gradient_checkpointing(accelerator)
    if accelerator.is_main_process:
        print(f"[Batch] per_rank={getattr(args, 'batch_size', 4)} world_size={accelerator.num_processes} "
              f"accumulation={accelerator.gradient_accumulation_steps}", flush=True)

    # (1) Seamless resume: load full state AFTER prepare so accelerator
    #     knows about the prepared optimizer / scheduler / dataloader.
    if resume_state_path:
        if not os.path.isdir(resume_state_path):
            raise SystemExit(
                f"[Resume-State] --resume_state path is not a directory: {resume_state_path}"
            )
        accelerator.load_state(resume_state_path)
        if accelerator.is_main_process:
            print(
                f"[Resume-State] restored from {resume_state_path} "
                f"(optimizer + LR scheduler + RNG + dataloader)",
                flush=True,
            )

    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                smoke_gradients.clear()
                unwrapped_model = accelerator.unwrap_model(model)
                if hasattr(unwrapped_model, "state_debug_current_step"):
                    unwrapped_model.state_debug_current_step = model_logger.num_steps + 1
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                if smoke and accelerator.sync_gradients:
                    if not torch.isfinite(loss).all():
                        raise RuntimeError("Smoke test produced a non-finite loss")
                    dit = unwrapped_model.pipe.dit
                    trainable_grads = sum(p.grad is not None for p in dit.parameters() if p.requires_grad)
                    frozen_grads = sum(p.grad is not None for p in dit.parameters() if not p.requires_grad)
                    if frozen_grads:
                        raise RuntimeError("Frozen parameters received gradients")
                    if smoke_hooks and (len(smoke_gradients) != len(smoke_hooks) or not all(
                            torch.isfinite(value) for value in smoke_gradients.values())):
                        raise RuntimeError("Missing or non-finite latent/tokenizer gradients")
                    print(f"[Smoke gradients] rank={accelerator.process_index} visible_grad_tensors={trainable_grads} "
                          f"frozen={frozen_grads} norms=" + str({
                              key: value.item() for key, value in smoke_gradients.items()}), flush=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if not accelerator.sync_gradients:
                    continue
                if smoke:
                    model_logger.num_steps += 1
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        print(f"[Smoke memory] rank={accelerator.process_index} step={model_logger.num_steps} "
                              f"loss={loss.item():.6f} peak_allocated_GiB={torch.cuda.max_memory_allocated()/2**30:.3f} "
                              f"peak_reserved_GiB={torch.cuda.max_memory_reserved()/2**30:.3f}", flush=True)
                else:
                    model_logger.on_step_end(accelerator, model, save_steps, loss=loss)

                # (2) Optionally dump full accelerator state alongside the
                #     step-<N>.safetensors written by model_logger above.
                if max_train_steps is not None and model_logger.num_steps >= max_train_steps:
                    if accelerator.is_main_process:
                        print(f"[Max-Steps] reached {max_train_steps}; stopping smoke run", flush=True)
                    if not smoke:
                        model_logger.on_training_end(accelerator, model, save_steps)
                    for hook in smoke_hooks:
                        hook.remove()
                    return

                if (
                    not smoke and save_full_state
                    and save_steps is not None
                    and model_logger.num_steps % save_steps == 0
                ):
                    state_dir = os.path.join(
                        model_logger.output_path, f"state-{model_logger.num_steps}"
                    )
                    if accelerator.is_main_process:
                        os.makedirs(state_dir, exist_ok=True)
                    accelerator.wait_for_everyone()
                    accelerator.save_state(state_dir)
                    if accelerator.is_main_process:
                        print(f"[Save-State] wrote {state_dir}", flush=True)

        if save_steps is None and not smoke:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    if not smoke:
        model_logger.on_training_end(accelerator, model, save_steps)
    for hook in smoke_hooks:
        hook.remove()

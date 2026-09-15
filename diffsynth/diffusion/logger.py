import gc, os, torch
from accelerate import Accelerator
from safetensors.torch import save_file


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        loss = kwargs.get("loss")
        if loss is not None and accelerator.is_main_process:
            print(f"[step {self.num_steps}] loss={loss.item():.6f}", flush=True)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = self._export_trainable_state_dict(accelerator, model)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            tmp_path = f"{path}.tmp"
            save_file(state_dict, tmp_path)
            os.replace(tmp_path, path)
            param_count = sum(tensor.numel() for tensor in state_dict.values())
            size_gib = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values()) / (1024 ** 3)
            print(
                f"[Checkpoint] saved {path} tensors={len(state_dict)} "
                f"params={param_count:,} size={size_gib:.2f} GiB",
                flush=True,
            )
            del state_dict
            gc.collect()
        accelerator.wait_for_everyone()


    def _export_trainable_state_dict(self, accelerator: Accelerator, model: torch.nn.Module):
        module = accelerator.unwrap_model(model)
        state_dict = {}
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            key = name
            if self.remove_prefix_in_ckpt is not None and key.startswith(self.remove_prefix_in_ckpt):
                key = key[len(self.remove_prefix_in_ckpt):]
            tensor = param.detach()
            if tensor.device.type == "cpu":
                tensor = tensor.clone()
            else:
                tensor = tensor.to(device="cpu")
            state_dict[key] = tensor.contiguous()
        return state_dict

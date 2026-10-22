"""One explicit compute backend for uninterrupted and restored training."""
import os
import torch

def describe():
    return {'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'gradient_reduction':'name-ordered-fp32-global-mean-v2','torch_version':torch.__version__}

def configure():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8',':16:8'):raise ValueError('Deterministic CUDA requires a supported CUBLAS_WORKSPACE_CONFIG')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    return describe()

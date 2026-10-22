"""Public checkout configuration must not depend on a private provider or root assets."""
from pathlib import Path
from rl_posttrain.config import load_config
from rl_posttrain.io import ROOT,digest


def test_environment_endpoint_overrides_inherited_templates(monkeypatch):
    monkeypatch.setenv("STATEPLAY_GPT_API_BASE_URL", "https://teacher.example.invalid/v1/")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://fallback.example.invalid/v1")
    config=load_config(ROOT/"rl_posttrain/configs/api_b_only_threaded.yaml")
    assert config["reward"]["endpoint"]=="https://teacher.example.invalid/v1"
    assert config["config_hash"]==digest({k:v for k,v in config.items() if k!="config_hash"})
    assert config["reward"]["concurrency_backend"]=="threads"


def test_public_paths_and_fallback_endpoint(monkeypatch):
    monkeypatch.delenv("STATEPLAY_GPT_API_BASE_URL",raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL",raising=False)
    config=load_config(ROOT/"rl_posttrain/configs/local_b_only.yaml")
    assert config["reward"]["endpoint"]=="https://api.example.invalid/v1"
    assert config["world"]["checkpoint"]==str(ROOT/"outputs/latent_world/step-4572.safetensors")
    assert config["reward"]["prompt"]==str(ROOT/"rl_posttrain/prompts/mechanics_reward_v2.txt")
    assert config["rollout"]["conditions_per_batch"]==28
    assert all(Path(p).is_relative_to(ROOT) for p in config["data"]["exclusion_archives"])


def test_complete_static_examples_are_packaged():
    from PIL import Image
    for name in ("macro_success", "result_win", "result_lose"):
        with Image.open(ROOT/"rl_posttrain/assets"/(name+".png")) as image:
            image.verify()

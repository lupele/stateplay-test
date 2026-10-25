# StatePlay DiffusionNFT reinforcement learning

This directory contains the three-stage post-training implementation for the implicit-state StatePlay architecture:

| Stage | Reward | Update |
|---|---|---|
| A | One API request scores four candidate videos with `gpt-6.1-sol` | DiffusionNFT world-model updates |
| B | Stage A's permanent videos and teacher scores | Full multimodal local Qwen3.5-9B language LoRA |
| C | Frozen Stage B reward model | Continued DiffusionNFT from Stage A |

B-only updates the latent-state transition module; Coupling-only updates coupling LoRA; Both updates their union. The remaining backbone and reference policy are protected. Checkpoints include the selected live/old/ref parameters, optimizer, RNG and update cursor. They require the original SFT and base models and are not standalone full DiT exports.

## Dependencies and local files

Use separate Python environments for WORLD and REWARD; their tested Torch/Transformers versions differ. The root `requirements.txt` lists shared dependencies. The two `.in` files below add compatible per-runtime pins, without publishing a dump of unrelated server packages.

```bash
python3.12 -m venv outputs/rl_posttrain/_envs/world
python3.12 -m venv outputs/rl_posttrain/_envs/reward
outputs/rl_posttrain/_envs/world/bin/python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
outputs/rl_posttrain/_envs/world/bin/python -m pip install -r rl_posttrain/requirements-world.in
outputs/rl_posttrain/_envs/reward/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
outputs/rl_posttrain/_envs/reward/bin/python -m pip install -r rl_posttrain/requirements-reward.in
```

Required assets are supplied separately:

- `outputs/latent_world/step-4572.safetensors`: original implicit SFT; its fixed identity is validated. Never overwrite this source file.
- `outputs/tokenizer/transition_tokenizer.pt`: trained transition tokenizer.
- `base_model/Wan-AI/Wan2.2-TI2V-5B/` and the Wan tokenizer paths used by the base project.
- `base_model/Qwen/Qwen3.5-9B/`: complete vision and language weights, processor and tokenizer. Loading uses `local_files_only=True`; no text-only fallback.
- `data/SF3/train/clips_5s/metadata_state_polish.csv` and its referenced videos/actions.
- Test exclusion archives under `data/test_archives/`, or explicit paths in a local configuration.

All paths in the public YAML templates are relative to the project root. The three static scoring examples are included in `rl_posttrain/assets/`, so publishing RL does not require changing the root `assets/` directory. Generated data, checkpoints, logs, credentials and Python environments are excluded from this code release.

## Private API configuration

No real relay address or credential is included. Configure `STATEPLAY_GPT_API_BASE_URL` (or `OPENAI_BASE_URL`) locally, and inject the key through `STATEPLAY_GPT_API_KEY` (or `OPENAI_API_KEY`). The YAML endpoint is deliberately a non-routable example. Environment overrides are applied before the configuration hash is calculated, so provider identity remains part of caching and checkpoint provenance. Never commit a credential file or an actual provider-specific configuration.

The teacher model remains `gpt-6.1-sol`. Each request contains four candidates, one real reference, 24 ordered frames per video, the initial observation and three static examples: 124 images total. The scorer checks exactly four finite scores and retains responses, cache, retry accounting and candidate mapping in local outputs. API retries are bounded and cannot guarantee exactly-once provider billing.

## Entry points

Run commands from the project root after supplying the models, data, local API configuration and required acceptance artifacts:

```bash
# A: API reward + NFT.
bash rl_posttrain/scripts/run_api.sh --run-id your_run --config rl_posttrain/configs/api_b_only_threaded.yaml
# B: train and evaluate the local reward model on permanent teacher data.
bash rl_posttrain/scripts/train_rm.sh --run-id your_run --config rl_posttrain/configs/rm_qwen35.yaml
# C: construct a configuration from the actual A checkpoint, then launch it.
outputs/rl_posttrain/_envs/world/bin/python -m rl_posttrain.prepare_local_config --run-id your_run --gpus 0,1,2,3 --conditions-per-batch 28 --generation-batch-size 1 --reward-batch-size 1
bash rl_posttrain/scripts/run_local.sh --run-id your_run --config PATH_PRINTED_ABOVE --gpus 0,1,2,3 --generation-batch-size 1 --reward-batch-size 1
```

`run_study_v1.sh` is the historical automatic A/B/C controller. It also needs the local API variables; its default provider-independent threaded configuration limits A to 2,280 conditions. Existing run state always takes precedence over starting a new traversal.

Formal launch requires source-bound software acceptance, and C additionally requires the independent reward-quality and real human-review gates. This repository does not ship those run-specific output files or pretend that a fresh checkout is already accepted. `--smoke` is for isolated software checks, not formal model-quality claims. The historical CPU/GPU diagnostic scripts may refer to source-server run IDs and are not automatic fresh-checkout launchers. In particular, inference-parity diagnostics require a `StatePlayPipeline` entry supporting implicit checkpoints; the RL trainer itself uses the implicit `diffsynth` adapter directly. This release does not replace the base repository's inference entry or independent evaluation tools.

## Stage C behavior

Generation, scoring and updating run sequentially; each phase uses all GPU IDs selected by `--gpus`. The previous phase exits before the next model is loaded. There are four progress bars: total committed conditions, generated videos, complete scoring groups and optimizer updates. ETA includes model switching and checkpoint saves.

- The ordinary default is **28 conditions / 112 candidate videos**, independent of GPU count.
- `--generation-batch-size` controls simultaneous candidates of one condition per GPU, from 1 to 4; every condition still has four distinct seeds.
- `--reward-batch-size` controls complete 124-image requests per GPU batch.
- `--gpus` supports a non-contiguous physical-device list.
- The initial C audit still uses 56 conditions. A+C traverse the original condition ledger once; B consumes no new world-model conditions. Tail batches use the actual remainder.
- Already generated videos, cached scores and committed updates are reused when resuming; changing GPU count in a partially updated batch is rejected.

More details: [C scheduling](docs/stage_c_timeslice_zh.md), [reward quality](docs/reward_model_quality.md), [game semantics](docs/game_rules_appendix.md). Test reports and actual training results belong in separately published artifacts; the code does not imply that A100 throughput or formal C quality has been measured.

## Tests and inference export

```bash
outputs/rl_posttrain/_envs/world/bin/python -m pytest rl_posttrain/tests -q
outputs/rl_posttrain/_envs/reward/bin/python -m pytest rl_posttrain/tests/test_timeslice.py -q
outputs/rl_posttrain/_envs/world/bin/python -m rl_posttrain.export_checkpoint --checkpoint /path/to/world.pt --output /path/to/StatePlay_NFT.safetensors
```

Unit tests use small CPU fixtures and mocked HTTP. Real-model smoke checks are separate and must use an isolated run/output directory. The full multimodal reward model, gradients, saving/reloading, and the real three-stage chain require the model files and GPUs above.

The NFT implementation follows official commit `cbb14f84b8312b620390dfcbe2fab69c1383b104`. Wan's decreasing sigma convention, clean first-frame masking, selected-parameter policy snapshots, EMA, and the optional SFT anchors are implemented in `algorithms/` and `train_world.py`.

### Stage C trainable modules and storage recovery

`launch_local` and the generated `start_c.sh` accept `--update-mode B-only`,
`--update-mode Coupling-only`, or `--update-mode Both` (`--train-mode` is an alias).
The default remains **B-only**. Existing GPU, global condition batch, generation
batch, reward batch, and progress bar options are unchanged.

```bash
bash stateplay/.stateplay_bootstrap/launch/start_c.sh \
  --gpus 4,5,6,7 --conditions-per-batch 28 \
  --generation-batch-size 1 --reward-batch-size 1 \
  --update-mode Coupling-only
```

Select modules before the first C reservation. The initializer preserves the
completed A model, its original old/ref policies, RNG and one-pass ledger; it
creates fresh Adam moments only for newly trainable Coupling LoRA parameters.
Frozen A transition parameters are stored in C checkpoints and included during
inference export. Resume with the same mode; changing an existing C branch is
refused. A conditions are excluded from C rollouts; the existing anchor sampler
and all reward/quality gates remain unchanged.

The standalone `bootstrap_stateplay.py` is now included at repository root.
Replace the older external bootstrap with this version before running the same
`--prepare-only` command. Recognized clean older checkouts can fast-forward;
modified or divergent checkouts are preserved. Previously prepared data,
checkpoints and cursors are retained, and a recognized old installed bootstrap
is backed up before refreshing its copy. `check.sh` still reports genuine
acceptance/quality blockers; this update does not manufacture passing gates.

Shared-storage path resolution, required-file checks and atomic JSON writes now
have bounded retries for ENOENT/EIO/ESTALE/ETIMEDOUT. A heartbeat metadata failure
does not orphan a live training child. Permanent absence, permission errors,
checksum failures and quality failures still stop safely. Appends and optimizer
updates are never blindly replayed by the storage retry helper.

Runtime `--conditions-per-batch` overrides also apply when resuming Coupling-only
or Both. The saved module-selection configuration and checkpoint stay immutable;
only subsequent rollout reservations use the requested size. A reserved batch
keeps its original conditions. Changing GPU count during a partially updated
batch is refused; finish that batch on its original topology first.

"""Train-only condition catalog. Mechanism class and NPC strategy are distinct."""
import collections
import hashlib
import math
from pathlib import Path
import shutil
import zipfile
import pandas as pd
import torch
from PIL import Image
from ..io import read_json, write_json, sha256, digest, save_tensors, ROOT
from .frames import video_info, loaded_frame_count, read_frames, jpeg_asset, extract_assets
from .provenance import rendering_identity

BUTTONS = ["UP", "DOWN", "LEFT", "RIGHT", "Y", "X", "Z", "A", "B", "C", "D"]
CLASSES = {"normal", "macro_success", "macro_fail", "result_win", "result_lose"}

def catalog(config, run):
    path = run / "conditions/catalog.json"
    source_hash = sha256(config["data"]["metadata"])
    if path.exists():
        existing = read_json(path)
        if existing["metadata_sha256"] != source_hash: raise RuntimeError("Training metadata changed; use a new run")
        if config['data'].get('split_policy')=='source-action-temporal-v2' and existing.get('split_policy')!='source-action-temporal-v2':raise RuntimeError('Legacy catalog lacks source/action/temporal leakage audit; use a new run')
        return existing["records"]
    root = Path(config["data"]["root"]); frame = pd.read_csv(config["data"]["metadata"])
    required = {"video", "action", "distribution", "prompt", "category"}
    if not required.issubset(frame.columns): raise ValueError(f"Missing metadata columns {required-set(frame.columns)}")
    excluded_hashes = set()
    for directory in config["data"].get("exclusion_roots", []):
        directory = Path(directory)
        if directory.is_dir() and directory.name != "evaluation":
            for video in directory.rglob("video.mp4"): excluded_hashes.add(sha256(video))
    # Supplied holdouts are independent of folder naming. Hash their raw video members.
    for archive in config["data"].get("exclusion_archives", []):
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if member.endswith("/video.mp4"):
                    with z.open(member) as f:
                        h = hashlib.sha256()
                        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
                    excluded_hashes.add(h.hexdigest())
    records = []; excluded = []; seen = set()
    for index, row in frame.iterrows():
        video = (root / row["video"]).resolve(); action = (root / row["action"]).resolve()
        if root.resolve() not in video.parents or root.resolve() not in action.parents: raise ValueError("Data path outside train root")
        if not video.exists() or not action.exists(): raise FileNotFoundError(f"Missing source record {index}")
        cls = str(row["distribution"]).strip().lower()
        if cls not in CLASSES: raise ValueError(f"Unknown mechanism class: {cls}")
        h = sha256(video)
        if h in excluded_hashes or h in seen:
            excluded.append({"source_row": int(index), "reason": "holdout_hash" if h in excluded_hashes else "duplicate_train_video", "sha256": h}); continue
        seen.add(h)
        episode = next((str(row[k]) for k in ("episode_id", "episode", "match_id") if k in row and pd.notna(row[k])), None)
        recording = next((str(row[k]) for k in ("source_recording", "recording_id", "source_video_id") if k in row and pd.notna(row[k])), None)
        # No fabricated episode IDs. A coarse visual cluster is conservative across action variations.
        if config['data'].get('split_policy')=='source-action-temporal-v2':
            visual_cluster=h[:16]  # Replaced by the full parallel temporal audit below.
        else:
            first = read_frames(video, 1)[0].resize((8, 8)).convert("RGB")
            visual_cluster = hashlib.sha256(bytes((int(v)//64 for pixel in first.getdata() for v in pixel))).hexdigest()[:16]
        records.append({"condition_id": f"train-{index:07d}", "source_row": int(index), "video": str(video), "action": str(action), "video_sha256": h, "action_sha256": sha256(action), "mechanism_class": cls, "npc_prompt": str(row["prompt"]), "npc_strategy": str(row["category"]), "episode_id": episode, "source_recording":recording, "cluster_id": "episode-" + episode if episode else "coarse-visual-" + visual_cluster})
    if config['data'].get('split_policy')=='source-action-temporal-v2':
        from .leakage import build_index
        index=build_index(records,config,run)
        bad=set(index['holdout_overlap_excluded'])
        excluded += [{'condition_id':r['condition_id'],'reason':'holdout_source_action_temporal_overlap'} for r in records if r['condition_id'] in bad]
        records=[dict(r,cluster_id=index['source_clusters'][r['condition_id']]) for r in records if r['condition_id'] not in bad]
    write_json(path, {"schema": "rl-plan-v1", "metadata_sha256": source_hash, "records": records, "excluded": excluded, 'split_policy':config['data'].get('split_policy','coarse-visual-v1'), "class_counts": dict(collections.Counter(r["mechanism_class"] for r in records)), "cluster_limitation": "Source/action/temporal components are conservative duplicate checks, not proof of episode independence; manual source review remains required."})
    return records

def action_data(path, requested=101, hold_window=10):
    frame = pd.read_parquet(path)
    if not set(BUTTONS).issubset(frame.columns): raise ValueError("Missing action columns; do not fabricate inputs")
    raw = torch.as_tensor(frame[BUTTONS].values[:requested].copy(), dtype=torch.float32)
    if not ((raw == 0) | (raw == 1)).all(): raise ValueError("Action buttons must be binary")
    indices = (torch.arange(len(raw)) // hold_window * hold_window).clamp(max=len(raw)-1)
    return raw, raw[indices].unsqueeze(0)

def bins(raw_length, latent_frames):
    return [{"start": math.floor(i * raw_length / latent_frames), "end_exclusive": math.ceil((i + 1)*raw_length/latent_frames)} for i in range(latent_frames)]

def prepare_group(record, config, run, old_version, counter):
    group_id = f"g-{counter:07d}-{record['condition_id']}"
    root = run / "groups" / group_id; path = root / "manifest.json"
    if path.exists():
        existing = read_json(path)
        if existing["condition_id"] != record["condition_id"] or existing["old_version"] != old_version: raise RuntimeError("Group policy/source mismatch")
        return root
    if sha256(record['video'])!=record['video_sha256'] or sha256(record['action'])!=record['action_sha256']:raise RuntimeError('Source video/actions changed after the immutable catalog was created')
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record["video"], root / "reference.mp4"); shutil.copy2(record["action"], root / "actions_raw.parquet")
    initial = read_frames(root / "reference.mp4", 1)[0]; initial.save(root / "initial.png")
    raw, dense = action_data(root / "actions_raw.parquet", config["rollout"]["requested_frames"], config["rollout"]["action_hold_window"])
    save_tensors(root / "actions_dense.safetensors", {"actions": dense})
    info = video_info(root / "reference.mp4"); loaded = loaded_frame_count(info["frames"], config["rollout"]["requested_frames"]); f = (loaded-1)//4+1
    condition = {**record, "available_frames": info["frames"], "loaded_frames": loaded, "latent_frames": f, "fps": info["fps"], "action_rows": len(raw), "action_bins": bins(len(raw), f), "button_order": BUTTONS}
    write_json(root / "condition.json", condition)
    candidates = []
    for slot in range(1, 5):
        seed = int(digest([config["plan"]["seed"], record["condition_id"], counter, slot])[:15], 16)
        candidates.append({"slot": f"video_{slot}", "video_id": f"v-{slot}-{seed}", "seed": seed, "old_version": old_version, "complete": False})
    if len({x["seed"] for x in candidates})!=4:raise RuntimeError("Candidate seed collision")
    write_json(path, {"schema": "rl-group-v1", "group_id": group_id, "condition_id": record["condition_id"], "old_version": old_version, "condition": condition, "candidates": candidates, "config_hash": config["config_hash"], "sft_sha256": config["world"]["checkpoint_sha256"], "tokenizer_sha256": sha256(config["world"]["tokenizer_checkpoint"]), "protocol": config["reward"]["protocol"], "prompt_sha256": sha256(config["reward"]["prompt"]), "rendering":rendering_identity(config), "latent_dtype": "bfloat16", "latent_scale": "Wan2.2_VAE normalized latent, final Euler output; no MP4 re-encoding"})
    return root

def complete_protocol(root, config):
    manifest = read_json(root / "manifest.json")
    if not all(c["complete"] for c in manifest["candidates"]): raise RuntimeError("Incomplete four-candidate group")
    assets = {}; reward = config["reward"]
    for name, video in [("reference_video", root / "reference.mp4")] + [(c["slot"], root / "candidates" / c["video_id"] / "video.mp4") for c in manifest["candidates"]]:
        assets[name] = extract_assets(video, root / "frames" / name, 24, reward["max_side"], reward["jpeg_quality"])
    assets["initial"] = [jpeg_asset(Image.open(root/"initial.png"), root/"frames/initial.jpg", reward["max_side"], reward["jpeg_quality"])]
    for name in ("macro_success", "result_win", "result_lose"):
        assets[name] = [jpeg_asset(Image.open(Path(__file__).resolve().parents[1]/"assets"/(name+".png")), root/"frames"/(name+".jpg"), reward["max_side"], reward["jpeg_quality"])]
    manifest["frame_assets"] = assets; manifest["frame_asset_hash"] = digest(assets)
    write_json(root/"manifest.json", manifest)
    return manifest

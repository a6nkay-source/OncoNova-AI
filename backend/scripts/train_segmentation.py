"""Train the demo 3-D U-Net on the synthetic phantom dataset.

    python scripts/train_segmentation.py --config ../configs/segmentation_unet_demo.yaml

Generates the dataset if its manifest is missing, trains with a patient-level
train/val split, keeps the checkpoint with the best validation Dice, and writes a
model card (with checkpoint hash and input profile) plus an experiment log.
The test split is never touched here; scripts/evaluate_segmentation.py uses it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onconova.config import get_settings  # noqa: E402
from onconova.data.manifest import build_synthetic_segmentation_dataset, load_manifest  # noqa: E402
from onconova.imaging.io import load_nifti  # noqa: E402
from onconova.imaging.normalize import intensity_profile, normalize_for_model  # noqa: E402
from onconova.segmentation.metrics import dice  # noqa: E402
from onconova.segmentation.registry import UNET_DIR_NAME  # noqa: E402
from onconova.segmentation.unet import NETWORK_KWARGS, build_network, pick_device, sha256_file  # noqa: E402


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).parent).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_split(manifest: dict, root: Path, split: str):
    items = []
    for r in manifest["records"]:
        if r["split"] != split:
            continue
        vol = load_nifti(root / r["image"], modality="CT")
        lab = load_nifti(root / r["lesions"], is_label=True).array
        x, _ = normalize_for_model(vol)
        items.append({"record": r, "volume": vol, "x": x, "y": lab})
    return items


def random_patch(item, patch, fg_prob, rng):
    x, y = item["x"], item["y"]
    shape = np.array(x.shape)
    patch = np.minimum(np.array(patch), shape)
    present = item.setdefault("classes", [int(c) for c in np.unique(y) if c > 0])
    if rng.random() < fg_prob and present:
        # Pick a class first so rare classes (node-region, other) are sampled as often as the primary.
        cls = present[rng.integers(len(present))]
        voxels = item.setdefault(f"fg_{cls}", np.argwhere(y == cls))
        center = voxels[rng.integers(len(voxels))]
        start = np.clip(center - patch // 2 + rng.integers(-8, 9, 3), 0, shape - patch)
    else:
        start = np.array([rng.integers(0, s - p + 1) for s, p in zip(shape, patch)])
    sl = tuple(slice(int(a), int(a + p)) for a, p in zip(start, patch))
    xp, yp = x[sl].copy(), y[sl].copy()
    if rng.random() < 0.5:  # left-right flip
        xp, yp = xp[:, :, ::-1].copy(), yp[:, :, ::-1].copy()
    xp = xp * rng.uniform(0.95, 1.05) + rng.uniform(-0.03, 0.03)
    return xp.astype(np.float32), yp.astype(np.int64)


def main() -> None:
    import torch
    from monai.losses import DiceCELoss

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/segmentation_unet_demo.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override for quick runs")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    tcfg, dcfg = cfg["training"], cfg["dataset"]
    settings = get_settings()

    data_root = settings.data_dir / "synthetic" / dcfg["manifest_id"]
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        print(f"Generating synthetic dataset at {data_root} ...", flush=True)
        build_synthetic_segmentation_dataset(
            data_root, n_patients=dcfg["n_patients"], seed=dcfg["dataset_seed"], split_seed=dcfg["split_seed"],
            timepoints_per_patient=dcfg["timepoints_per_patient"], manifest_id=dcfg["manifest_id"])
    manifest = load_manifest(manifest_path)

    seed = tcfg["seed"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = pick_device() if tcfg["device"] == "auto" else tcfg["device"]
    train, val = load_split(manifest, data_root, "train"), load_split(manifest, data_root, "val")
    print(f"train scans={len(train)} val scans={len(val)} device={device}", flush=True)

    net = build_network().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(net.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    epochs = args.epochs or tcfg["epochs"]
    steps_per_epoch = max(1, len(train) // tcfg["batch_size"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exp_dir = settings.artifacts_dir / "experiments" / f"{cfg['experiment']}-{run_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    log = (exp_dir / "metrics.jsonl").open("w")
    out_dir = settings.models_dir / UNET_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    best, best_epoch, t0 = -1.0, -1, time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        order = rng.permutation(len(train))
        for step in range(steps_per_epoch):
            batch = [random_patch(train[i], tcfg["patch_zyx"], tcfg["foreground_crop_probability"], rng)
                     for i in order[step * tcfg["batch_size"]:(step + 1) * tcfg["batch_size"]]]
            x = torch.from_numpy(np.stack([b[0] for b in batch])[:, None]).to(device)
            y = torch.from_numpy(np.stack([b[1] for b in batch])[:, None]).to(device)
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.detach().cpu()))
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "elapsed_s": round(time.time() - t0, 1)}
        if epoch % tcfg["validate_every"] == 0 or epoch == epochs:
            net.eval()
            scores_primary, scores_any = [], []
            with torch.no_grad():
                for item in val:
                    x = torch.from_numpy(item["x"][None, None]).to(device)
                    pred = net(x).argmax(1)[0].cpu().numpy()
                    scores_primary.append(dice(pred == 1, item["y"] == 1))
                    scores_any.append(dice(pred > 0, item["y"] > 0))
            entry["val_dice_primary"] = float(np.mean(scores_primary))
            entry["val_dice_any_lesion"] = float(np.mean(scores_any))
            if entry["val_dice_primary"] > best:
                best, best_epoch = entry["val_dice_primary"], epoch
                torch.save(net.state_dict(), out_dir / "model.pt")
        print(json.dumps(entry), flush=True)
        log.write(json.dumps(entry) + "\n")
        log.flush()
    log.close()

    train_profiles = [intensity_profile(item["volume"]) for item in train]
    spacings = np.array([item["volume"].spacing for item in train])
    card = {
        "version": "0.1.0",
        "trained_on": (f"{len(train)} synthetic CT phantom scans from "
                       f"{manifest['patient_counts_by_split']['train']} synthetic patients "
                       f"(manifest {manifest['manifest_id']}, generator {manifest['generator']})."),
        "training_manifest": manifest["manifest_id"],
        "checkpoint_sha256": sha256_file(out_dir / "model.pt"),
        "network": {k: list(v) if isinstance(v, tuple) else v for k, v in NETWORK_KWARGS.items()},
        "best_epoch": best_epoch,
        "best_val_dice_primary": best,
        "experiment_dir": str(exp_dir.relative_to(settings.artifacts_dir)),
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_profile": {
            "source": "synthetic",
            "modality": "CT",
            "spacing_min": spacings.min(0).round(3).tolist(),
            "spacing_max": spacings.max(0).round(3).tolist(),
            **{f"{k}_range": [round(min(p[k] for p in train_profiles), 1), round(max(p[k] for p in train_profiles), 1)]
               for k in ("p1", "p50", "p99")},
        },
        "config": cfg,
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    (exp_dir / "summary.json").write_text(json.dumps({"best_epoch": best_epoch, "best_val_dice_primary": best}, indent=2))
    print(f"Saved {out_dir / 'model.pt'} (best val primary Dice {best:.3f} at epoch {best_epoch})")


if __name__ == "__main__":
    main()

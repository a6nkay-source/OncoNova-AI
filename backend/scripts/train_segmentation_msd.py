"""Train the 3-D U-Net on real CT: the MSD Task06 Lung training patients.

    python scripts/train_segmentation_msd.py --config ../configs/segmentation_unet_msd.yaml [--seed 1] [--epochs N]

1. Cache: each training patient's scan is normalised, resampled to the model grid
   (linear; the mask by partial-volume fraction >= 0.5), cropped to the body and
   stored as .npy, memory-mapped during training. Only ``train`` patients are read:
   ``val`` patients are kept for the model comparison and ``test`` stays locked.
2. Checkpoint selection uses an inner, patient-level split of the train patients.
3. One model per seed, the checkpoint with the best inner-validation Dice. Each gets
   a model card (hash, grid, input profile, patients used); an ensemble card lists
   the seeds that exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onconova.config import get_settings  # noqa: E402
from onconova.data.manifest import load_manifest  # noqa: E402
from onconova.data.splits import patient_level_split  # noqa: E402
from onconova.imaging.io import load_nifti  # noqa: E402
from onconova.imaging.normalize import MODEL_CT_RANGE, intensity_profile, normalize_for_model  # noqa: E402
from onconova.segmentation.grid import model_grid, to_grid  # noqa: E402
from onconova.segmentation.metrics import dice  # noqa: E402
from onconova.segmentation.unet import NETWORK_KWARGS, TILE_OVERLAP, build_network, pick_device, sha256_file  # noqa: E402

from train_segmentation import git_commit  # noqa: E402

BODY_HU = -500.0
LABEL = "RESEARCH MODEL - trained on public lung CT (MSD Task06, 44 patients); not validated for clinical use."


def body_crop(x: np.ndarray, label: np.ndarray, margin: int = 4) -> tuple[slice, ...]:
    """Bounding box of the body (largest region above -500 HU) and the tumour, plus a margin."""
    lo, hi = MODEL_CT_RANGE
    body = x > (BODY_HU - lo) / (hi - lo)
    comps, n = ndimage.label(body)
    if n:
        body = comps == (int(np.argmax(ndimage.sum(body, comps, range(1, n + 1)))) + 1)
    pts = np.argwhere(body | (label > 0))
    a = np.maximum(pts.min(0) - margin, 0)
    b = np.minimum(pts.max(0) + margin + 1, x.shape)
    return tuple(slice(int(i), int(j)) for i, j in zip(a, b))


def build_cache(records: list[dict], root: Path, cache_dir: Path, spacing: tuple[float, float, float]) -> list[dict]:
    items = []
    for r in records:
        assert r["split"] == "train", "only training patients are cached"
        out = cache_dir / r["patient_id"]
        if not (out / "meta.json").exists():
            t0 = time.time()
            vol = load_nifti(root / r["image"], modality="CT")
            lab = load_nifti(root / r["lesions"], is_label=True)
            x, _ = normalize_for_model(vol)
            grid = model_grid(vol, spacing)
            xg = to_grid(x, vol, grid).astype(np.float32)
            yg = (to_grid((lab.array > 0).astype(np.float32), vol, grid) >= 0.5).astype(np.uint8)
            sl = body_crop(xg, yg)
            xg, yg = xg[sl], yg[sl]
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "image.npy", xg.astype(np.float16))
            np.save(out / "label.npy", yg)
            np.save(out / "foreground.npy", np.argwhere(yg > 0).astype(np.int32))
            profile = intensity_profile(vol)
            (out / "meta.json").write_text(json.dumps({
                "scan_id": r["scan_id"], "source_shape_zyx": list(vol.shape), "source_spacing_xyz": list(vol.spacing),
                "grid_spacing_xyz": list(spacing), "crop_start_zyx": [s.start for s in sl], "shape_zyx": list(xg.shape),
                "tumour_voxels": int(yg.sum()), "intensity_profile": profile}))
            print(f"  cached {r['patient_id']} {list(vol.shape)} -> {list(xg.shape)} ({time.time() - t0:.0f} s)", flush=True)
        meta = json.loads((out / "meta.json").read_text())
        items.append({"patient_id": r["patient_id"], "meta": meta,
                      "x": np.load(out / "image.npy", mmap_mode="r"), "y": np.load(out / "label.npy", mmap_mode="r"),
                      "fg": np.load(out / "foreground.npy")})
    return items


def random_patch(item: dict, patch: np.ndarray, fg_prob: float, rng: np.random.Generator):
    x, y = item["x"], item["y"]
    shape = np.array(x.shape)
    size = np.minimum(patch, shape)
    if rng.random() < fg_prob and len(item["fg"]):
        centre = item["fg"][rng.integers(len(item["fg"]))]
        start = np.clip(centre - size // 2 + rng.integers(-16, 17, 3), 0, shape - size)
    else:
        start = np.array([rng.integers(0, s - p + 1) for s, p in zip(shape, size)])
    sl = tuple(slice(int(a), int(a + p)) for a, p in zip(start, size))
    xp, yp = np.asarray(x[sl], dtype=np.float32), np.asarray(y[sl], dtype=np.int64)
    pad = [(0, int(p - s)) for p, s in zip(patch, xp.shape)]  # scans smaller than a patch
    if any(b for _, b in pad):
        xp, yp = np.pad(xp, pad, mode="edge"), np.pad(yp, pad)
    if rng.random() < 0.5:  # left-right flip: lung tumours occur on either side
        xp, yp = xp[:, :, ::-1], yp[:, :, ::-1]
    xp = xp * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05)
    return np.ascontiguousarray(xp), np.ascontiguousarray(yp)


def validate(net, items: list[dict], patch: tuple[int, int, int], device: str) -> list[float]:
    import torch
    from monai.inferers import sliding_window_inference

    net.eval()
    scores = []
    with torch.no_grad():
        for item in items:
            x = torch.from_numpy(np.asarray(item["x"], dtype=np.float32)[None, None]).to(device)
            logits = sliding_window_inference(x, roi_size=patch, sw_batch_size=2, predictor=net, overlap=TILE_OVERLAP,
                                              mode="gaussian", padding_mode="replicate")
            pred = logits.argmax(1)[0].cpu().numpy()
            scores.append(dice(pred > 0, np.asarray(item["y"]) > 0))
    return scores


def train_one(seed: int, cfg: dict, fit: list[dict], inner: list[dict], manifest: dict, epochs: int, iterations: int,
              settings) -> Path:
    import torch
    from monai.losses import DiceCELoss

    tcfg, mcfg = cfg["training"], cfg["model"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = pick_device() if tcfg["device"] == "auto" else tcfg["device"]
    network = {**{k: list(v) if isinstance(v, tuple) else v for k, v in NETWORK_KWARGS.items()}, **mcfg["network"]}
    net = build_network(network).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(net.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * iterations)
    patch = np.array(tcfg["patch_zyx"])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exp_dir = settings.artifacts_dir / "experiments" / f"{cfg['experiment']}-s{seed}-{run_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    out_dir = settings.models_dir / f"seg-unet-msd-s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"seed {seed}: fit patients={len(fit)} inner-val patients={len(inner)} device={device}", flush=True)

    best, best_epoch, t0 = -1.0, -1, time.time()
    with (exp_dir / "metrics.jsonl").open("w") as log:
        for epoch in range(1, epochs + 1):
            net.train()
            losses = []
            for _ in range(iterations):
                batch = [random_patch(fit[i], patch, tcfg["foreground_crop_probability"], rng)
                         for i in rng.integers(0, len(fit), tcfg["batch_size"])]
                x = torch.from_numpy(np.stack([b[0] for b in batch])[:, None]).to(device)
                y = torch.from_numpy(np.stack([b[1] for b in batch])[:, None]).to(device)
                opt.zero_grad()
                loss = loss_fn(net(x), y)
                loss.backward()
                opt.step()
                sched.step()
                losses.append(float(loss.detach().cpu()))
            entry = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), "elapsed_s": round(time.time() - t0, 1)}
            if epoch % tcfg["validate_every"] == 0 or epoch == epochs:
                scores = validate(net, inner, tuple(tcfg["patch_zyx"]), device)
                entry["inner_val_dice"] = round(float(np.mean(scores)), 4)
                entry["inner_val_dice_per_patient"] = [round(v, 3) for v in scores]
                if entry["inner_val_dice"] > best:
                    best, best_epoch = entry["inner_val_dice"], epoch
                    torch.save(net.state_dict(), out_dir / "model.pt")
            print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()

    profiles = [it["meta"]["intensity_profile"] for it in fit]
    spacings = np.array([it["meta"]["source_spacing_xyz"] for it in fit])
    card = {
        "version": "0.1.0",
        "card": {
            "id": f"{mcfg['model_id']}-s{seed}",
            "name": f"3-D U-Net trained on MSD Task06 Lung CT (seed {seed})",
            "label": LABEL,
            "classes": {"1": "primary_tumor"},
            "intended_use": "Research comparison of lung-tumour segmentation on public CT; not for diagnosis.",
            "limitations": [
                f"Trained on {len(fit)} patients of one public dataset (MSD Task06, TCIA-derived); other scanners, "
                "protocols and populations are unvalidated.",
                "Its class 1 is MSD's single 'cancer' label (the annotated tumour); it does not separate primary "
                "tumour, nodes or other lesions.",
                "Reference masks come from a single rater and are not pathology-confirmed.",
                f"Works on a {tcfg['spacing_xyz']} mm grid; tumours smaller than a few millimetres can be lost.",
                "Uncertainty from MC dropout reflects model variance only; it is not calibrated risk.",
            ],
        },
        "trained_on": (f"{len(fit)} real CT scans from {len(fit)} patients of MSD Task06 Lung "
                       f"({manifest['source'].get('licence')}); {len(inner)} more training patients for checkpoint selection."),
        "training_manifest": manifest["manifest_id"],
        "checkpoint_sha256": sha256_file(out_dir / "model.pt"),
        "network": network,
        "inference": {"spacing_xyz": tcfg["spacing_xyz"], "patch_zyx": tcfg["patch_zyx"], "overlap": TILE_OVERLAP,
                      "blending": "gaussian"},
        "best_epoch": best_epoch,
        "best_inner_val_dice": best,
        "patients": {"fit": [it["patient_id"] for it in fit], "inner_val": [it["patient_id"] for it in inner]},
        "experiment_dir": str(exp_dir.relative_to(settings.artifacts_dir)),
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_profile": {
            "source": manifest["manifest_id"],
            "modality": "CT",
            "spacing_min": spacings.min(0).round(3).tolist(),
            "spacing_max": spacings.max(0).round(3).tolist(),
            **{f"{k}_range": [round(min(p[k] for p in profiles), 1), round(max(p[k] for p in profiles), 1)]
               for k in ("p1", "p50", "p99")},
        },
        "seed": seed,
        "config": cfg,
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    (exp_dir / "summary.json").write_text(json.dumps({"best_epoch": best_epoch, "best_inner_val_dice": best}, indent=2))
    print(f"Saved {out_dir / 'model.pt'} (best inner-val Dice {best:.3f} at epoch {best_epoch}, {time.time() - t0:.0f} s)")
    return out_dir


def write_ensemble_card(settings, cfg: dict) -> Path | None:
    members = sorted(settings.models_dir.glob("seg-unet-msd-s*/model_card.json"))
    if len(members) < 2:
        return None
    cards = [json.loads(p.read_text()) for p in members]
    ens = {k: cards[0][k] for k in ("version", "network", "inference", "input_profile", "training_manifest")}
    ens["card"] = {**cards[0]["card"], "id": f"{cfg['model']['model_id']}-ens",
                   "name": f"Ensemble of {len(cards)} 3-D U-Nets trained on MSD Task06 Lung CT (seeds "
                           f"{', '.join(str(c['seed']) for c in cards)})"}
    ens["trained_on"] = cards[0]["trained_on"] + f" Ensemble of {len(cards)} seeds; probabilities averaged."
    ens["members"] = [{"file": f"../{p.parent.name}/model.pt", "sha256": c["checkpoint_sha256"], "seed": c["seed"]}
                      for p, c in zip(members, cards)]
    ens["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = settings.models_dir / "seg-unet-msd-ens"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_card.json").write_text(json.dumps(ens, indent=2))
    print(f"Ensemble card for {len(cards)} seeds -> {out / 'model_card.json'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/segmentation_unet_msd.yaml")
    parser.add_argument("--seed", type=int, default=None, help="train only this seed")
    parser.add_argument("--epochs", type=int, default=None, help="override for quick runs")
    parser.add_argument("--iterations", type=int, default=None, help="override iterations per epoch")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    settings = get_settings()
    manifest = load_manifest(settings.data_dir / "external" / cfg["dataset"]["manifest_id"] / "manifest.json")
    root = Path(manifest["root"])
    tcfg = cfg["training"]
    spacing = tuple(float(v) for v in tcfg["spacing_xyz"])

    train_records = [r for r in manifest["records"] if r["split"] == "train"]
    inner_split = patient_level_split([r["patient_id"] for r in train_records], seed=cfg["dataset"]["inner_split_seed"],
                                      fractions=(1 - cfg["dataset"]["inner_val_fraction"], cfg["dataset"]["inner_val_fraction"], 0.0))
    cache_dir = settings.data_dir / "external" / cfg["dataset"]["manifest_id"] / f"cache-{'x'.join(str(v) for v in spacing)}"
    print(f"Caching {len(train_records)} training patients at {spacing} mm in {cache_dir}", flush=True)
    items = build_cache(train_records, root, cache_dir, spacing)
    fit = [it for it in items if inner_split[it["patient_id"]] == "train"]
    inner = [it for it in items if inner_split[it["patient_id"]] == "val"]

    seeds = [args.seed] if args.seed is not None else tcfg["seeds"]
    for seed in seeds:
        train_one(seed, cfg, fit, inner, manifest, args.epochs or tcfg["epochs"], args.iterations or tcfg["iterations_per_epoch"],
                  settings)
    write_ensemble_card(settings, cfg)


if __name__ == "__main__":
    main()

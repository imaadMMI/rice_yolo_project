"""
Rice leaf disease classification on Dhan-Shomadhan with Ultralytics YOLO (classification head).

CSCI435/935 group project - reference implementation.

Pipeline
    1. prepare : read ./Dhan-Shomadhan, fix folder-name typos, letterbox every image to a square
                 (keeps the whole tall leaf) and write a resized cache under the dataset folder.
    2. run     : for 5 random stratified 70/10/20 splits, fine-tune a COCO/ImageNet-pretrained YOLO
                 classifier and evaluate it on the white / field / combined test images.
    3. compare : one-split pilot that ranks several YOLO versions/sizes (used to pick the model).
    4. predict : classify new images with a trained checkpoint.

Usage (see README.md for details)
    python yolo.py prepare
    python yolo.py run                         # all experiments, 5 runs, default model
    python yolo.py compare --models yolov8s-cls.pt yolo11s-cls.pt yolo26s-cls.pt
    python yolo.py predict --weights <best.pt> --source <image or folder>
"""

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "Dhan-Shomadhan"

# Folder names in the dataset contain typos and differ between the two backgrounds.
CLASS_FIX = {
    "Browon Spot": "Brown_Spot", "Brown Spot": "Brown_Spot",
    "Leaf Scaled": "Leaf_Scald",
    "Rice Blast": "Rice_Blast",
    "Rice Turgro": "Rice_Tungro", "Rice Tungro": "Rice_Tungro",
    "Sheath Blight": "Sheath_Blight", "Shath Blight": "Sheath_Blight",
}
BG_FIX = {"White Background": "white", "Field Background": "field"}
CLASSES = sorted(set(CLASS_FIX.values()))
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
PAD_VALUE = 114  # YOLO letterbox grey

# Hyper-parameters chosen from the literature (see README) and a pilot on split 0.
DEFAULTS = dict(
    model="yolov8s-cls.pt",
    imgsz=512,
    epochs=60,
    patience=15,
    batch=32,
    lr0=5e-4,
    dropout=0.2,
    runs=5,
    seed=42,
    cache_size=640,
)


# ----------------------------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------------------------
def imread(path):
    # np.fromfile + imdecode handles non-ASCII paths on Windows; IMREAD_COLOR honours EXIF rotation
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def letterbox_square(img, size):
    """Resize so the long side == size, then pad the short side to make a size x size image."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (size - nh) // 2, (size - nw) // 2
    return cv2.copyMakeBorder(img, top, size - nh - top, left, size - nw - left,
                              cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)


def scan_dataset(data_dir):
    """Return a list of dicts {src, cls, bg, name} for every image in the raw dataset."""
    items = []
    for bg_folder, bg in BG_FIX.items():
        for cls_dir in sorted((data_dir / bg_folder).iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in CLASS_FIX:
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    items.append(dict(src=f, cls=CLASS_FIX[cls_dir.name], bg=bg, name=f"{bg}_{f.stem}.jpg"))
    if not items:
        raise SystemExit(f"No images found under {data_dir}. Expected 'White Background' and 'Field Background'.")
    return items


def prepare(data_dir, work_dir, cache_size, force=False):
    """Build the letterboxed image cache: <work_dir>/cache/<bg>/<class>/<bg>_<name>.jpg"""
    items = scan_dataset(data_dir)
    cache = work_dir / "cache"
    manifest = work_dir / "manifest.json"
    if manifest.exists() and not force:
        meta = json.loads(manifest.read_text())
        if meta.get("cache_size") == cache_size and len(meta["items"]) == len(items):
            print(f"Cache already built ({len(items)} images) at {cache}")
            return meta["items"]

    print(f"Building {cache_size}x{cache_size} letterboxed cache for {len(items)} images -> {cache}")
    out = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        dst = cache / it["bg"] / it["cls"] / it["name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = imread(it["src"])
        if img is None:
            print(f"  ! unreadable, skipped: {it['src']}")
            continue
        cv2.imwrite(str(dst), letterbox_square(img, cache_size), [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append(dict(path=str(dst), cls=it["cls"], bg=it["bg"], src=str(it["src"])))
        if i % 200 == 0:
            print(f"  {i}/{len(items)}")
    manifest.write_text(json.dumps(dict(cache_size=cache_size, items=out), indent=1))
    print(f"Done in {time.time() - t0:.0f}s")
    return out


def make_split(items, run, seed, test_frac=0.2, val_frac=0.1):
    """Stratified (class x background) train/val/test split for one run."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(items))
    strata = [f"{it['cls']}|{it['bg']}" for it in items]
    rs = seed + run
    trval, test = train_test_split(idx, test_size=test_frac, stratify=strata, random_state=rs)
    tr, val = train_test_split(trval, test_size=val_frac / (1 - test_frac),
                               stratify=[strata[i] for i in trval], random_state=rs)
    return {"train": tr.tolist(), "val": val.tolist(), "test": test.tolist()}


def link_or_copy(src, dst):
    try:
        os.link(src, dst)  # hard link: no extra disk space
    except OSError:
        shutil.copy2(src, dst)


def materialise_split(items, split, bgs, out_dir):
    """Write an Ultralytics ImageFolder tree (train/val/test/<class>/img) for the given backgrounds."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for part, ids in split.items():
        for c in CLASSES:
            (out_dir / part / c).mkdir(parents=True, exist_ok=True)
        for i in ids:
            it = items[i]
            if it["bg"] in bgs:
                src = Path(it["path"])
                link_or_copy(src, out_dir / part / it["cls"] / src.name)
    return out_dir


# ----------------------------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------------------------
def pick_device():
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"Using GPU: {name} ({mem:.1f} GB), CUDA {torch.version.cuda}, torch {torch.__version__}")
        return 0
    print("CUDA not available - training on CPU (this will be slow).")
    return "cpu"


def load_weights(name, weights_dir):
    """Download (once) the pretrained checkpoint into weights_dir and return a YOLO model."""
    from ultralytics import YOLO

    weights_dir.mkdir(parents=True, exist_ok=True)
    p = Path(name)
    return YOLO(str(p if p.exists() else weights_dir / p.name))


def free_trainer(model):
    """Shut down the trainer's dataloader workers and release GPU memory before the next model.
    On Windows each worker is a full Python process; left alive they pile up across trainings."""
    import torch

    t = getattr(model, "trainer", None)
    if t is not None:
        for name in ("train_loader", "test_loader"):
            loader = getattr(t, name, None)
            it = getattr(loader, "iterator", None)
            if it is not None and hasattr(it, "_shutdown_workers"):
                it._shutdown_workers()
            setattr(t, name, None)
        model.trainer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_one(args, data_dir, out_dir, name, seed, device):
    model = load_weights(args.model, args.work_dir / "weights")
    model.train(
        data=str(data_dir), project=str(out_dir), name=name, exist_ok=True,
        epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=args.batch,
        device=device, workers=args.workers, cache=False, amp=True,
        seed=seed, deterministic=True,
        # optimisation: AdamW + warm-up + cosine decay (Niaz et al. / Nggego et al.)
        optimizer="AdamW", lr0=args.lr0, lrf=0.01, cos_lr=True, warmup_epochs=3, weight_decay=5e-4,
        dropout=args.dropout,
        # augmentation: flips, colour jitter, scale crop, RandAugment, random erasing
        fliplr=0.5, flipud=0.5, hsv_h=0.015, hsv_s=0.4, hsv_v=0.4, scale=0.4,
        auto_augment="randaugment", erasing=0.3,
        plots=True, verbose=False,
    )
    free_trainer(model)
    best = out_dir / name / "weights" / "best.pt"
    last = best.with_name("last.pt")
    if last.exists():
        last.unlink()  # only keep the best checkpoint
    return best


def predict_probs(model, paths, imgsz, device, tta=False, batch=32):
    """Return (N, C) class probabilities ordered like CLASSES, and mean ms/image."""
    names = model.names
    order = [list(names.values()).index(c) for c in CLASSES]
    probs, t0 = [], time.perf_counter()
    for i in range(0, len(paths), batch):
        imgs = [imread(p) for p in paths[i:i + batch]]
        views = [imgs] + ([[cv2.flip(im, 1) for im in imgs], [cv2.flip(im, 0) for im in imgs]] if tta else [])
        acc = 0
        for v in views:
            res = model.predict(v, imgsz=imgsz, device=device, verbose=False)
            acc = acc + np.stack([r.probs.data.cpu().numpy() for r in res])
        probs.append(acc / len(views))
    ms = (time.perf_counter() - t0) * 1000 / max(len(paths), 1)
    return np.concatenate(probs)[:, order], ms


def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASSES)),
                                                 average="macro", zero_division=0)
    return dict(acc=accuracy_score(y_true, y_pred) * 100, precision=p * 100, recall=r * 100, f1=f * 100)


# Experiments: training backgrounds -> list of (scenario, test backgrounds)
EXPERIMENTS = {
    "combined": (("white", "field"), [("a_white", ("white",)), ("b_field", ("field",)), ("c_both", ("white", "field"))]),
    "white_only": (("white",), [("a_white", ("white",)), ("b_field_xdomain", ("field",))]),
    "field_only": (("field",), [("b_field", ("field",)), ("a_white_xdomain", ("white",))]),
}


def evaluate(model, items, test_ids, scenarios, imgsz, device, tta):
    test_items = [items[i] for i in test_ids]
    probs, ms = predict_probs(model, [it["path"] for it in test_items], imgsz, device, tta)
    y_true = np.array([CLASSES.index(it["cls"]) for it in test_items])
    y_pred = probs.argmax(1)
    bgs = np.array([it["bg"] for it in test_items])
    out, rows = {}, []
    for scen, sbgs in scenarios:
        m = np.isin(bgs, sbgs)
        out[scen] = dict(metrics(y_true[m], y_pred[m]), n=int(m.sum()), y_true=y_true[m].tolist(),
                         y_pred=y_pred[m].tolist())
    for it, t, p, pr in zip(test_items, y_true, y_pred, probs):
        if t != p:
            rows.append(dict(src=it["src"], bg=it["bg"], true=CLASSES[t], pred=CLASSES[p], conf=float(pr[p])))
    return out, rows, ms


def save_confusion(y_true, y_pred, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", ax=ax, xticks_rotation=30, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def run_experiments(args, items, device):
    from ultralytics import YOLO

    out_root = args.work_dir / "runs" / Path(args.model).stem
    out_root.mkdir(parents=True, exist_ok=True)
    results = {e: {} for e in args.experiments}
    misclassified = []
    for run in range(args.runs):
        split = make_split(items, run, args.seed)
        print(f"\n=== Run {run + 1}/{args.runs}  (train {len(split['train'])}, val {len(split['val'])}, "
              f"test {len(split['test'])}) ===")
        for exp in args.experiments:
            train_bgs, scenarios = EXPERIMENTS[exp]
            data_dir = materialise_split(items, split, train_bgs, args.work_dir / "splits" / exp)
            t0 = time.time()
            best = train_one(args, data_dir, out_root / exp, f"run{run + 1}", args.seed + run, device)
            train_s = time.time() - t0
            model = YOLO(str(best))
            res, wrong, ms = evaluate(model, items, split["test"], scenarios, args.imgsz, device, args.tta)
            for scen, r in res.items():
                results[exp].setdefault(scen, []).append(dict(r, run=run + 1, train_s=train_s, ms_per_img=ms))
                print(f"  [{exp:<10}] {scen:<16} acc {r['acc']:6.2f}%  F1 {r['f1']:6.2f}%  (n={r['n']})")
            misclassified += [dict(w, experiment=exp, run=run + 1) for w in wrong]
            print(f"  [{exp:<10}] trained in {train_s / 60:.1f} min, inference {ms:.1f} ms/img")
            del model
            gc.collect()
        (out_root / "results.json").write_text(json.dumps(results, indent=1))  # checkpoint after each run

    summarise(results, misclassified, out_root, args)
    return results


def summarise(results, misclassified, out_root, args):
    import csv

    lines = [f"# Results - {Path(args.model).stem}, imgsz {args.imgsz}, {args.runs} runs (mean ± std)", "",
             "| Experiment | Scenario | n test | Accuracy (%) | Macro-F1 (%) | Macro-Precision (%) | Macro-Recall (%) |",
             "|---|---|---|---|---|---|---|"]
    rows = []
    for exp, scens in results.items():
        for scen, runs in scens.items():
            agg = {k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs]))
                   for k in ("acc", "f1", "precision", "recall")}
            n = int(np.mean([r["n"] for r in runs]))
            lines.append(f"| {exp} | {scen} | {n} | " + " | ".join(f"{agg[k][0]:.2f} ± {agg[k][1]:.2f}"
                         for k in ("acc", "f1", "precision", "recall")) + " |")
            for r in runs:
                rows.append(dict(experiment=exp, scenario=scen, run=r["run"], n=r["n"],
                                 **{k: round(r[k], 3) for k in ("acc", "f1", "precision", "recall")},
                                 train_min=round(r["train_s"] / 60, 2), ms_per_img=round(r["ms_per_img"], 2)))
            # confusion matrix accumulated over all runs + per-class accuracy
            yt = sum((r["y_true"] for r in runs), [])
            yp = sum((r["y_pred"] for r in runs), [])
            cm = save_confusion(yt, yp, f"{exp} / {scen} (sum of {len(runs)} runs)",
                                out_root / f"cm_{exp}_{scen}.png")
            per_cls = cm.diagonal() / np.maximum(cm.sum(1), 1) * 100
            lines.append("|  |  |  | " + ", ".join(f"{c}: {a:.1f}" for c, a in zip(CLASSES, per_cls)) + " | | | |")

    t = [r for r in rows if r["experiment"] == "combined"] or rows
    lines += ["", f"Mean training time per model: {np.mean([r['train_min'] for r in t]):.1f} min; "
                  f"inference {np.mean([r['ms_per_img'] for r in t]):.1f} ms/image (incl. JPEG decode)."]
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf8")
    with open(out_root / "results_per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if misclassified:
        with open(out_root / "misclassified.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(misclassified[0]))
            w.writeheader()
            w.writerows(misclassified)
    print("\n" + "\n".join(lines))
    print(f"\nSaved summary, per-run CSV, confusion matrices and misclassified list to {out_root}")


def compare_models(args, items, device):
    """Pilot: train each candidate on a held-out pilot split (not one of the 5 reported splits) with the
    combined model and rank by accuracy on that split's test set."""
    from ultralytics import YOLO

    split = make_split(items, 100, args.seed)
    data_dir = materialise_split(items, split, ("white", "field"), args.work_dir / "splits" / "compare")
    table = []
    for m in args.models:
        for sz in args.sizes:
            a = argparse.Namespace(**{**vars(args), "model": m, "imgsz": sz})
            t0 = time.time()
            best = train_one(a, data_dir, args.work_dir / "compare", f"{Path(m).stem}_{sz}", args.seed, device)
            train_s = time.time() - t0
            model = YOLO(str(best))
            res, _, ms = evaluate(model, items, split["test"], EXPERIMENTS["combined"][1], sz, device, args.tta)
            row = dict(model=Path(m).stem, imgsz=sz, params_M=round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
                       **{s: round(r["acc"], 2) for s, r in res.items()},
                       f1_both=round(res["c_both"]["f1"], 2), train_min=round(train_s / 60, 1), ms_per_img=round(ms, 1))
            table.append(row)
            print(row)
    table.sort(key=lambda r: (-r["c_both"], -r["b_field"]))
    out = args.work_dir / "compare" / "compare.md"
    hdr = list(table[0])
    out.write_text("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" +
                   "\n".join("| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in table), encoding="utf8")
    print("\n" + out.read_text())


def predict_cli(args, device):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    src = Path(args.source)
    paths = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    tmp = []
    for p in paths:  # apply the same letterbox preprocessing used in training
        tmp.append(letterbox_square(imread(p), args.cache_size))
    for p, img in zip(paths, tmp):
        r = model.predict(img, imgsz=args.imgsz, device=device, verbose=False)[0]
        top = r.probs.top1
        print(f"{p.name:<40} {model.names[top]:<15} {float(r.probs.top1conf) * 100:5.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "run", "compare", "predict"])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="path to Dhan-Shomadhan folder")
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="interim data / outputs (default: <data>/_yolo, as required by the brief)")
    ap.add_argument("--model", default=DEFAULTS["model"], help="Ultralytics classification checkpoint")
    ap.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    ap.add_argument("--lr0", type=float, default=DEFAULTS["lr0"])
    ap.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    ap.add_argument("--runs", type=int, default=DEFAULTS["runs"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers (each is a process on Windows)")
    ap.add_argument("--cache-size", type=int, default=DEFAULTS["cache_size"])
    ap.add_argument("--tta", action="store_true", help="average predictions over h/v flips")
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    ap.add_argument("--models", nargs="+", default=["yolov8s-cls.pt", "yolo11s-cls.pt", "yolo26s-cls.pt"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[DEFAULTS["imgsz"]])
    ap.add_argument("--weights", help="checkpoint for predict")
    ap.add_argument("--source", help="image file or folder for predict")
    ap.add_argument("--force", action="store_true", help="rebuild the image cache")
    args = ap.parse_args()

    args.data = args.data.resolve()
    args.work_dir = (args.work_dir or args.data / "_yolo").resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare":
        prepare(args.data, args.work_dir, args.cache_size, args.force)
        return

    device = pick_device()
    # Ultralytics downloads helper weights (e.g. for the AMP check) into the CWD; keep them in work_dir.
    (args.work_dir / "weights").mkdir(exist_ok=True)
    os.chdir(args.work_dir / "weights")

    if args.command == "predict":
        predict_cli(args, device)
        return
    items = prepare(args.data, args.work_dir, args.cache_size)
    if args.command == "compare":
        compare_models(args, items, device)
    else:
        run_experiments(args, items, device)


if __name__ == "__main__":
    main()

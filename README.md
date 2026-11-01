# YOLOv8s Experiment: How to Run

This is the exploratory YOLOv8s experiment for the CSCI435/CSCI935 group project on rice leaf disease classification. It is separate from the proposed EfficientNet-B0 method in `group_6.py`.

Running it trains YOLOv8s on five random splits and tests each model on white-background, field-background and combined images.

## 1. Set up the folder

Place the dataset folder next to `yolo.py`:

```text
project-folder/
├── yolo.py
└── Dhan-Shomadhan/
    ├── Field Background/
    └── White Background/
```

Keep the dataset's original folder names. `yolo.py` corrects the spelling errors in them automatically.

The dataset is available from [Mendeley Data](https://data.mendeley.com/datasets/znsxdctwtt).

## 2. Install the packages

Use Python 3.12, ideally in a new virtual environment. This is the setup used for the reported results, on an NVIDIA GPU:

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install ultralytics==8.4.182 scikit-learn
```

Without an NVIDIA GPU, use this first line instead. This setup was not tested, and training will be much slower:

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

The pretrained `yolov8s-cls.pt` weights download automatically on first use, so no manual download is needed. An internet connection is required for that first run.

## 3. Check that it works (optional, about 3 minutes)

Run these from the folder containing `yolo.py`:

```bash
python yolo.py prepare
python yolo.py run --runs 1 --epochs 1 --experiments combined
```

This only tests the setup, so its low accuracy is expected. These results are saved to the same folder as the full run, and the full run replaces them.

## 4. Reproduce the results

```bash
python yolo.py prepare
python yolo.py run
```

- `prepare` builds a resized image cache. It took about 80 seconds in testing, and it skips this step if the cache already exists.
- `run` trains 15 models: three models on each of five splits. Expect about 2 to 4 hours on a mid-range GPU.

If you run out of memory:
- For system memory, add `--workers 1`. This is recommended on Windows computers with 16 GB of RAM.
- For GPU memory, add `--batch 16`. This setting differs from the reported experiment and may change the results.

## 5. Check the output

Everything is written under the dataset folder, in `Dhan-Shomadhan/_yolo/`. The results are in `Dhan-Shomadhan/_yolo/runs/yolov8s-cls/`:

| File | Contents |
|---|---|
| `summary.md` | Mean ± standard deviation for every experiment, with per-class accuracy |
| `results_per_run.csv` | Results for each of the five runs |
| `cm_*.png` | Confusion matrices summed over the five runs |
| `misclassified.csv` | Each incorrectly classified test image |

Compare `summary.md` with these reported accuracies (%):

| Training images | White test images | Field test images | Combined test images |
|---|---|---|---|
| Both backgrounds | 81.17 ± 0.41 | 70.00 ± 7.00 | 77.75 ± 2.38 |
| White only | 81.04 ± 2.11 | 34.41 ± 4.80 | Not tested |
| Field only | 32.21 ± 8.13 | 63.24 ± 4.26 | Not tested |

The original output files are in [`results/yolo/`](results/yolo/).

The data splits are fixed, and training uses fixed seeds with deterministic mode. Small differences are still possible on different hardware or package versions.

## Experiment settings

| Setting | Value |
|---|---|
| Model | `yolov8s-cls.pt`, pretrained on ImageNet |
| Train/validation/test split | 70/10/20 (773/111/222 images), stratified by disease and background |
| Split seeds | 42 to 46 |
| Input size | 512 × 512, padded to a square |
| Epochs | Up to 60, stopping after 15 without validation improvement |
| Batch size | 32 |
| Optimiser | AdamW, initial learning rate 5 × 10⁻⁴, cosine decay, 3 warm-up epochs |
| Weight decay and dropout | 5 × 10⁻⁴ and 0.2 |
| Checkpoint selection | Best validation score (mean of top-1 and top-5 accuracy) |

## Model pilot (optional)

YOLOv8s at 512 × 512 was chosen in a pilot on one separate split. That split overlaps the five evaluation splits. To repeat the pilot:

```bash
python yolo.py compare --models yolov8n-cls.pt yolov8s-cls.pt yolov8m-cls.pt --sizes 384 512
python yolo.py compare --models yolo11s-cls.pt yolo26s-cls.pt --sizes 512
```

Each command replaces `Dhan-Shomadhan/_yolo/compare/compare.md`, so copy that file between the two commands. The original results are in [`results/yolo/compare_all.md`](results/yolo/compare_all.md).

## Reported environment

| Package | Version |
|---|---|
| Python | 3.12.15 |
| PyTorch | 2.14.1 (CUDA 13.0) |
| torchvision | 0.29.1 |
| Ultralytics | 8.4.182 |
| OpenCV | 5.0.0 |
| NumPy | 2.5.2 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |

The reported experiments ran on an NVIDIA RTX 3060 Laptop GPU with 6 GB of memory.

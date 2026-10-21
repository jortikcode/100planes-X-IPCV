# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Image Processing and Computer Vision - Assignment Module #2
#
#
# Alessandro Teodori (0001247539) alessandro.teodori3@studio.unibo.it
#
# Juan Guillermo Jaramillo Saa (0001217725) juan.jaramillosaa@studio.unibo.it
#
# Ubalducci Michele (0001226810) michele.ubalducci@studio.unibo.it
#
# # Part 2B: tweaking the fine-tuning recipe of a pretrained ResNet-18
#
# Starting from the baseline fine-tuning of Part 2A (`torchvision.models.resnet18(weights=IMAGENET1K_V1)` trained with
# the from-scratch recipe of Part 1), we systematically adapt the training hyperparameters to the transfer-learning
# regime. Following the methodology of Part 1, we organize the study as a **ladder**: each step modifies **one aspect
# of the training recipe at a time** on top of the best configuration of the previous step, so that every change in
# validation accuracy has a single experimental cause.
#
# | step | what we tune | question |
# |---|---|---|
# | **s1: lr x weight decay** | grid search over smaller peak learning rates (`0.01` to `0.001`) and smaller $L_2$ weight decays (`5e-3` to `1e-4`) | how much accuracy do we recover once the update size and weight decay no longer erase the ImageNet initialization? |
# | **s2: head warmup & schedule** | freeze the feature extractor for the first $K \in \{0, 3, 5, 10\}$ epochs (training only `fc`), and compare `onecycle` vs. `cosine` decay with warmup | does aligning the random linear head before unfreezing the backbone reduce early feature distortion? |
# | **s3: progressive LRs & freezing** | layer-wise learning-rate decay across stages ($\text{lr}_s = \text{lr} \cdot \gamma^{5-s}$, $\gamma \in \{1.0, 0.7, 0.5, 0.3\}$) and permanent freezing of the lowest stages (`0, 1, 2`) | should early generic layers learn more slowly (or stay fixed) compared to late task-specific layers? |
# | **s4: data augmentation** | 4A: multi-scale crop (`0.25` to `0.5` min area) x random erasing (`0` to `0.5`); 4B: color jitter, Mixup and CutMix on GPU | does stronger augmentation limit rapid memorization of the 3,334 training images once pretrained features are preserved? |
# | **s5: classifier regularization & seeds** | recalibrate label smoothing (`0` to `0.2`) and dropout (`0` to `0.4`), confirm the top 2 recipes on three seeds (`42, 43, 44`), and train the winner for 150 epochs | how much head regularization is needed on top of a tuned backbone, and how does the final model compare to Part 1 and Part 2A? |
#
# **Protocol.** All exploration runs (steps s1–s5) use a budget of **70 epochs** (seed 42), matching the exploration
# budget of Part 1. The top two configurations at the end of the ladder are confirmed over three seeds (`42, 43, 44`),
# and the selected configuration is finally trained for **150 epochs** (three seeds) for a direct comparison with
# Part 1 and Part 2A. **All choices are made on the validation split; test accuracy is only reported.**

# %% [markdown]
# ## Setup
#
# **How to run.** The notebook runs top to bottom on Colab (GPU runtime). All runs are stored in `results/part2B` and
# downloaded automatically from the repository: a run whose JSON already exists with the same configuration is skipped,
# reproducing all tables and plots without retraining. Setting `RETRAIN = True` trains everything from scratch. Stored
# runs from `results/part2A` and `results/part1` are also downloaded for reference comparisons.

# %%
# %pip install -q torchinfo kagglehub

# %%
import contextlib, json, math, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops import roi_align
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

IN_COLAB = "google.colab" in sys.modules
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

# Mixed precision on CUDA only: bfloat16 where supported (A100, L4), float16 + loss scaling otherwise (T4)
AMP = DEVICE.type == "cuda"
AMP_DTYPE = torch.bfloat16 if (AMP and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
torch.set_float32_matmul_precision("high")

RETRAIN = False        # True: retrain runs even if their JSON already exists
SHOW_SUMMARY = True    # print the model summary when a run starts with a different architecture than the last one
USE_DRIVE = True      # Colab only: keep results and image cache in Google Drive (False: Colab's temporary disk)
print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name()})" if AMP else "")
      + f" | amp: {AMP_DTYPE if AMP else 'off'} | torch {torch.__version__}")

# %% [markdown]
# ### Paths
#
# Same layout as Part 1 and Part 2A: results are stored in `results/part2B`, while reference runs are read from
# `results/part2A` and `results/part1`. The preprocessed image cache in `CACHE_DIR` is shared across all parts.

# %%
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS_DIR = Path("/content/drive/MyDrive/ipcv/results/part2B")
    PART2A_DIR = Path("/content/drive/MyDrive/ipcv/results/part2A")
    PART1_DIR = Path("/content/drive/MyDrive/ipcv/results/part1")
    CACHE_DIR = Path("/content/drive/MyDrive/ipcv/cache")
    DATA_ROOT = Path("/content/data")
elif IN_COLAB:
    RESULTS_DIR = Path("/content/results/part2B")
    PART2A_DIR = Path("/content/results/part2A")
    PART1_DIR = Path("/content/results/part1")
    CACHE_DIR = Path("/content/cache")
    DATA_ROOT = Path("/content/data")
else:
    REPO_DIR = Path.cwd().parent if Path.cwd().name == "code" else Path.cwd()
    RESULTS_DIR = REPO_DIR / "results" / "part2B"
    PART2A_DIR = REPO_DIR / "results" / "part2A"
    PART1_DIR = REPO_DIR / "results" / "part1"
    DATA_ROOT = Path.home() / "data" / "fgvc"
    CACHE_DIR = DATA_ROOT / "cache"

for d in (RESULTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
print("results ->", RESULTS_DIR)

GITHUB_REPO, GITHUB_REF = "jortikcode/100planes-X-IPCV", "main"
FETCH_RESULTS = IN_COLAB


def fetch_results(repo, ref, dest, part):
    # Download the repo as one archive and copy its results/<part> JSON files into dest, never overwriting.
    import io, tarfile, urllib.request
    prefix = f"results/{part}/"
    with urllib.request.urlopen(f"https://github.com/{repo}/archive/{ref}.tar.gz") as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
    added = kept = 0
    for member in archive.getmembers():
        path = member.name.split("/", 1)[-1]
        name = path.rsplit("/", 1)[-1]
        if (not (member.isfile() and path.startswith(prefix) and name.endswith(".json"))
                or ".." in path or name.startswith(".")):
            continue
        target = dest / path[len(prefix):]
        if target.exists():
            kept += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.extractfile(member).read())
        added += 1
    print(f"results/{part} from github.com/{repo} ({ref}): {added} downloaded, {kept} already present")


if FETCH_RESULTS:
    for part, dest in (("part2B", RESULTS_DIR), ("part2A", PART2A_DIR), ("part1", PART1_DIR)):
        try:
            fetch_results(GITHUB_REPO, GITHUB_REF, dest, part)
        except Exception as e:
            print(f"WARNING: could not download results/{part} ({e})")

# %% [markdown]
# ### Reproducibility

# %%
def fix_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


fix_random(42)

# %% [markdown]
# ## Configuration
#
# `BASE` extends the dictionary of Part 2A with the hyperparameters explored in Part 2B, keeping backward-compatible
# defaults (`freeze_epochs=0`, `freeze_stages=0`, `layer_decay=1.0`, `color_jitter=0.0`, `mixup_alpha=0.0`,
# `cutmix_alpha=0.0`). `START_2B` is the starting point of the 2B ladder: the Part 1 / Part 2A recipe evaluated at the
# 70-epoch exploration budget.
#
# | parameter | explored values | motivation & reference |
# |---|---|---|
# | `lr` | `0.01, 0.005, 0.002, 0.001` | pretrained weights already lie in a good basin; a smaller step size prevents catastrophic forgetting (Yosinski et al., NeurIPS 2014; Karpathy, CS231n) |
# | `weight_decay` | `5e-3, 1e-3, 5e-4, 1e-4` | standard $L_2$ penalty pulls weights toward 0 rather than toward the ImageNet initialization $\theta_0$ (Li et al., ICML 2018) |
# | `freeze_epochs` | `0, 3, 5, 10` | Linear Probing then Fine-Tuning (LP-FT): warming up the random `fc` head while the extractor is frozen prevents early gradient distortion (Kumar et al., ICLR 2022; Kilcher, 2022) |
# | `schedule` | `onecycle`, `cosine` (3 warmup epochs) | tests whether a rapid warmup + cosine decay preserves pretrained features better than a long 30% LR ramp |
# | `layer_decay` | `1.0, 0.7, 0.5, 0.3` | discriminative / progressive fine-tuning: stage $s$ receives $\text{lr} \cdot \gamma^{5-s}$ so generic low-level filters change slower than high-level layers (Howard & Ruder, ACL 2018; Howard, fast.ai) |
# | `freeze_stages` | `0, 1, 2` | permanently freezes `stem` (`1`) or `stem + layer1` (`2`) to preserve universal edge/texture detectors |
# | `crop_scale`, `random_erasing` | `(0.5,1.0), (0.35,1.0), (0.25,1.0)` x `0.0, 0.25, 0.5` | stronger scale variation and random occlusion force the network to use multiple aircraft parts (Zhong et al., AAAI 2020) |
# | `color_jitter`, `mixup_alpha`, `cutmix_alpha` | jitter `0.2`, Mixup `0.2`, CutMix `0.5, 1.0` | batch-level mixing regularizes confident pretrained representations on small fine-grained datasets (Zhang et al., ICLR 2018; Yun et al., ICCV 2019) |
# | `label_smoothing`, `dropout` | `0.0, 0.05, 0.1, 0.2` x `0.0, 0.2, 0.4` | recalibrates classifier regularization once backbone plasticity and augmentation are tuned (Szegedy et al., CVPR 2016; Srivastava et al., JMLR 2014) |

# %%
BASE = dict(
    seed=42,
    # data & augmentation
    img_size=224,
    augment=True,
    crop_scale=(0.5, 1.0),
    random_erasing=0.25,
    color_jitter=0.0,
    mixup_alpha=0.0,
    cutmix_alpha=0.0,
    # architecture & freezing
    dropout=0.0,
    freeze_epochs=0,
    freeze_stages=0,
    # optimization & schedule
    epochs=70,
    batch_size=64,
    lr=0.05,
    layer_decay=1.0,
    momentum=0.9,
    weight_decay=5e-3,
    label_smoothing=0.2,
    warmup_epochs=3,
    schedule="onecycle",
)

START_2B = dict(
    augment=True,
    crop_scale=(0.5, 1.0),
    dropout=0.0,
    epochs=70,
    lr=0.05,
    weight_decay=5e-3,
    label_smoothing=0.2,
    random_erasing=0.25,
    schedule="onecycle",
)

SEEDS = [42, 43, 44]
TOP_K = 2
LONG_EPOCHS = 150

# Step 1 search space: learning rate x weight decay
S1_LRS = [0.01, 0.005, 0.002, 0.001]
S1_WDS = [5e-3, 1e-3, 5e-4, 1e-4]

# Step 2 search space: frozen-backbone warmup epochs x learning-rate schedule
S2_FREEZE_EPOCHS = [0, 3, 5, 10]
S2_SCHEDULES = ["onecycle", "cosine"]

# Step 3 search space: progressive layer-wise LR decay x permanently frozen lower stages
S3_DECAYS = [1.0, 0.7, 0.5, 0.3]
S3_FREEZE_STAGES = [0, 1, 2]

# Step 4 search space: 4A spatial crop scale x random erasing; 4B photometric jitter & batch mixing
S4_CROPS = [(0.5, 1.0), (0.35, 1.0), (0.25, 1.0)]
S4_ERASING = [0.0, 0.25, 0.5]
S4_MIX_CONFIGS = [
    ("mix_none",     dict(color_jitter=0.0, mixup_alpha=0.0, cutmix_alpha=0.0)),
    ("mix_jitter02", dict(color_jitter=0.2, mixup_alpha=0.0, cutmix_alpha=0.0)),
    ("mix_mixup02",  dict(color_jitter=0.0, mixup_alpha=0.2, cutmix_alpha=0.0)),
    ("mix_cutmix05", dict(color_jitter=0.0, mixup_alpha=0.0, cutmix_alpha=0.5)),
    ("mix_cutmix10", dict(color_jitter=0.0, mixup_alpha=0.0, cutmix_alpha=1.0)),
]

# Step 5 search space: label smoothing x classifier dropout
S5_SMOOTH = [0.0, 0.05, 0.1, 0.2]
S5_DROPOUT = [0.0, 0.2, 0.4]

# %% [markdown]
# ## Data and GPU Augmentation
#
# Same cached `uint8` tensors on GPU and ImageNet normalization (`mean = (0.485, 0.456, 0.406)`,
# `std = (0.229, 0.224, 0.225)`) as Part 2A. To test stronger augmentations in Step 4 without CPU bottlenecks, we add
# vectorized GPU implementations of brightness/contrast jitter (`gpu_color_jitter`), **Mixup** (Zhang et al., ICLR 2018)
# and **CutMix** (Yun et al., ICCV 2019).

# %%
BANNER_PX = 20
KAGGLE_MIRROR = "seryouxblaster764/fgvc-aircraft"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def fetch_from_kaggle(root):
    import kagglehub
    path = Path(kagglehub.dataset_download(KAGGLE_MIRROR))
    source = next((p for p in [path, *path.rglob("fgvc-aircraft-2013b")] if (p / "data" / "images").is_dir()), None)
    if source is None:
        raise RuntimeError(f"unexpected layout of the Kaggle mirror in {path}")
    target = root / "fgvc-aircraft-2013b"
    root.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if (target / "data" / "images").is_dir():
            return
        raise RuntimeError(f"{target} exists but is incomplete: remove it and run again")
    target.symlink_to(source, target_is_directory=True)


def build_cache(split, img_size):
    path = CACHE_DIR / f"{split}_{img_size}_stretch.pt"
    legacy = CACHE_DIR / f"{split}_{img_size}.pt"
    for p in (path, legacy):
        if p.exists():
            return torch.load(p)
    try:
        dset = datasets.FGVCAircraft(DATA_ROOT, split=split, annotation_level="variant", download=True)
    except Exception as official_error:
        print(f"official download failed ({official_error}): using the Kaggle mirror {KAGGLE_MIRROR}")
        try:
            fetch_from_kaggle(DATA_ROOT)
            dset = datasets.FGVCAircraft(DATA_ROOT, split=split, annotation_level="variant", download=True)
        except Exception as e:
            raise RuntimeError(f"could not download FGVC-Aircraft, neither from the official server "
                               f"({official_error}) nor from Kaggle ({e})") from e
    images, labels = [], []
    for i in tqdm(range(len(dset)), desc=f"caching {split} @ {img_size}px"):
        img, label = dset[i]
        img = img.crop((0, 0, img.width, img.height - BANNER_PX))
        images.append(TF.pil_to_tensor(TF.resize(img, [img_size, img_size], antialias=True)))
        labels.append(label)
    cached = (torch.stack(images), torch.tensor(labels), dset.classes)
    torch.save(cached, path)
    return cached


class GPUSplit:
    def __init__(self, split, img_size):
        x, y, self.classes = build_cache(split, img_size)
        self.x, self.y = x.to(DEVICE), y.to(DEVICE)

    def __len__(self):
        return len(self.y)


_DATA = {}


def get_data(img_size):
    if img_size not in _DATA:
        splits = {s: GPUSplit(s, img_size) for s in ("train", "val", "test")}
        mean = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
        _DATA[img_size] = dict(splits, mean=mean, std=std)
    return _DATA[img_size]


def gpu_augment(x, size, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)):
    # RandomResizedCrop + horizontal flip on a float batch, a different crop for every image.
    B, _, H, W = x.shape
    dev = x.device
    area = torch.empty(B, device=dev).uniform_(*scale) * H * W
    log_r = torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]))
    r = torch.exp(log_r)
    w = torch.sqrt(area * r).clamp(max=W)
    h = torch.sqrt(area / r).clamp(max=H)
    x0 = torch.rand(B, device=dev) * (W - w)
    y0 = torch.rand(B, device=dev) * (H - h)
    idx = torch.arange(B, device=dev, dtype=x.dtype)
    boxes = torch.stack([idx, x0, y0, x0 + w, y0 + h], dim=1)
    out = roi_align(x, boxes, output_size=size, aligned=True)
    flip = torch.rand(B, device=dev) < 0.5
    return torch.where(flip.view(B, 1, 1, 1), out.flip(-1), out)


def gpu_color_jitter(x, cj):
    # Fast per-image brightness and contrast jitter on a float batch in [0, 1].
    B = x.shape[0]
    dev = x.device
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(1.0 - cj, 1.0 + cj)
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(1.0 - cj, 1.0 + cj)
    mean = x.mean(dim=(2, 3), keepdim=True)
    return ((x - mean) * c + mean * b).clamp(0.0, 1.0)


def gpu_random_erasing(x, p, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
    # RandomErasing on a normalized batch: with probability p, a random rectangle of each image is set to 0.
    B, _, H, W = x.shape
    dev = x.device
    area = torch.empty(B, device=dev).uniform_(*scale) * H * W
    r = torch.exp(torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1])))
    h = torch.sqrt(area * r).clamp(max=H).view(B, 1, 1)
    w = torch.sqrt(area / r).clamp(max=W).view(B, 1, 1)
    y0 = torch.rand(B, 1, 1, device=dev) * (H - h)
    x0 = torch.rand(B, 1, 1, device=dev) * (W - w)
    ys = torch.arange(H, device=dev).view(1, H, 1)
    xs = torch.arange(W, device=dev).view(1, 1, W)
    mask = (ys >= y0) & (ys < y0 + h) & (xs >= x0) & (xs < x0 + w)
    mask &= (torch.rand(B, 1, 1, device=dev) < p)
    return x.masked_fill(mask.unsqueeze(1), 0.0)


def gpu_mixup_cutmix(x, y, cfg):
    # Batch-level Mixup or CutMix on GPU. Returns (x_mixed, y_a, y_b, lam).
    B, _, H, W = x.shape
    if cfg.get("cutmix_alpha", 0.0) > 0.0:
        alpha = cfg["cutmix_alpha"]
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(B, device=x.device)
        cut_rat = math.sqrt(1.0 - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = random.randint(0, W), random.randint(0, H)
        x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
        x2, y2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1)) / (W * H)
        return x, y, y[perm], lam
    if cfg.get("mixup_alpha", 0.0) > 0.0:
        alpha = cfg["mixup_alpha"]
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(B, device=x.device)
        x = lam * x + (1.0 - lam) * x[perm]
        return x, y, y[perm], lam
    return x, y, y, 1.0


def batches(split, data, cfg, train):
    n = len(split)
    order = torch.randperm(n).to(DEVICE) if train else torch.arange(n, device=DEVICE)
    bs = cfg["batch_size"] if train else 512
    for i in range(0, n, bs):
        idx = order[i:i + bs]
        if train and len(idx) < bs:
            break
        x = split.x[idx].float() / 255
        if train and cfg["augment"]:
            x = gpu_augment(x, cfg["img_size"], scale=tuple(cfg["crop_scale"]))
        if train and cfg.get("color_jitter", 0.0) > 0.0:
            x = gpu_color_jitter(x, cfg["color_jitter"])
        x = (x - data["mean"]) / data["std"]
        if train and cfg["random_erasing"] > 0:
            x = gpu_random_erasing(x, cfg["random_erasing"])
        yield x.contiguous(memory_format=torch.channels_last), split.y[idx]


try:
    data = get_data(BASE["img_size"])
except Exception as e:
    data = None
    print(f"WARNING: the dataset is not available ({e})\n"
          "The cells that need the images are skipped; the stored results are still loaded and shown.")

N_CLASSES = 100
if data is not None:
    CLASSES = data["train"].classes
    assert len(CLASSES) == N_CLASSES
    print(f"classes: {N_CLASSES} | train {len(data['train'])} | val {len(data['val'])} | test {len(data['test'])}")

# %% [markdown]
# ## The network and stage decomposition
#
# We decompose the pretrained ResNet-18 (`ResNet18_Weights.IMAGENET1K_V1`) into its 6 sequential stages:
#
# ```
# stage 0 (stem)     conv1 (7x7, s2) + bn1 + maxpool (s2)    224 -> 56, 64 ch     lr * layer_decay^5
# stage 1 (layer1)   2 x BasicBlock(64)                       56x56               lr * layer_decay^4
# stage 2 (layer2)   2 x BasicBlock(128), first stride 2      28x28               lr * layer_decay^3
# stage 3 (layer3)   2 x BasicBlock(256), first stride 2      14x14               lr * layer_decay^2
# stage 4 (layer4)   2 x BasicBlock(512), first stride 2       7x7                lr * layer_decay^1
# stage 5 (fc)       Dropout(p) -> Linear(512, 100)            1x1                lr * 1.0
# ```
#
# This decomposition supports both **progressive learning rates** (`layer_decay` $\gamma \le 1$) and **stage freezing**:
# - during the initial `freeze_epochs`, stages 0–4 are frozen and only stage 5 (`fc`) is updated;
# - throughout training, stages $< \text{freeze\_stages}$ remain frozen.
#
# Whenever a stage is frozen, its `BatchNorm2d` layers are placed in `.eval()` mode during `train_epoch` so that its
# pretrained ImageNet running mean and variance are not altered while its weights are fixed.

# %%
def build_model(cfg, n_classes=N_CLASSES):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(model.fc.in_features, n_classes))
    return model


def get_stages(model):
    # The 6 sequential stage modules of ResNet-18, from input stem (0) to classifier head (5).
    return [
        ("stem",   nn.ModuleList([model.conv1, model.bn1])),
        ("layer1", model.layer1),
        ("layer2", model.layer2),
        ("layer3", model.layer3),
        ("layer4", model.layer4),
        ("fc",     model.fc),
    ]


def apply_freezing(model, cfg, epoch):
    # Set requires_grad and BatchNorm eval mode for frozen stages at the given epoch.
    warmup_freeze = epoch < cfg.get("freeze_epochs", 0)
    freeze_stages = cfg.get("freeze_stages", 0)
    for idx, (_, mod) in enumerate(get_stages(model)):
        frozen = False if idx == 5 else (warmup_freeze or (idx < freeze_stages))
        mod.requires_grad_(not frozen)
        if frozen:
            mod.eval()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_gflops(model, img_size):
    macs = 0

    def hook(m, inp, out):
        nonlocal macs
        if isinstance(m, nn.Conv2d):
            macs += out[0].numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            macs += m.in_features * m.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model.eval()(torch.zeros(1, 3, img_size, img_size))
    for h in handles:
        h.remove()
    return 2 * macs / 1e9


ARCH_KEYS = ("img_size", "dropout", "freeze_stages")


def show_summary(cfg, depth=4):
    m = build_model(cfg)
    apply_freezing(m, cfg, epoch=cfg.get("freeze_epochs", 0))
    return torchinfo.summary(m, input_size=(1, 3, cfg["img_size"], cfg["img_size"]),
                             col_names=("output_size", "num_params", "trainable"), row_settings=("var_names",),
                             depth=depth, device="cpu", verbose=0)

# %% [markdown]
# ## Training
#
# - **Optimizer**: SGD with momentum 0.9, split into per-stage parameter groups with `lr_scale = layer_decay ** (5 - s)`
#   and weight decay excluded on BatchNorm parameters and biases (`p.ndim <= 1`).
# - **Schedule**: `onecycle` (30% cosine ramp, 70% cosine decay) or `cosine` (linear warmup for `warmup_epochs`, then
#   cosine decay to 0).
# - **Loss**: cross-entropy with label smoothing (mixed with $\lambda$ when Mixup or CutMix is active). Validation loss
#   is always plain unweighted cross-entropy.

# %%
def autocast():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if AMP else contextlib.nullcontext()


def make_optimizer(model, cfg):
    groups = []
    ld = cfg.get("layer_decay", 1.0)
    for idx, (stage_name, mod) in enumerate(get_stages(model)):
        scale = ld ** (5 - idx)
        decay = [p for p in mod.parameters() if p.ndim > 1]
        no_decay = [p for p in mod.parameters() if p.ndim <= 1]
        if decay:
            groups.append({"params": decay, "weight_decay": cfg["weight_decay"],
                           "lr_scale": scale, "stage_idx": idx, "stage": stage_name})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0,
                           "lr_scale": scale, "stage_idx": idx, "stage": stage_name})
    return torch.optim.SGD(groups, lr=cfg["lr"], momentum=cfg["momentum"])


def lr_at(epoch, cfg):
    if cfg.get("schedule", "cosine") == "onecycle":
        up = max(round(0.3 * cfg["epochs"]), 1)
        low, high, end = cfg["lr"] / 25, cfg["lr"], cfg["lr"] / 1e4
        if epoch < up:
            return high + (low - high) * (1 + math.cos(math.pi * epoch / up)) / 2
        t = (epoch - up) / max(cfg["epochs"] - up, 1)
        return end + (high - end) * (1 + math.cos(math.pi * t)) / 2
    if epoch < cfg["warmup_epochs"]:
        return cfg["lr"] * (epoch + 1) / cfg["warmup_epochs"]
    progress = (epoch - cfg["warmup_epochs"]) / max(cfg["epochs"] - cfg["warmup_epochs"], 1)
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, opt, scaler, criterion, data, cfg, epoch):
    model.train()
    apply_freezing(model, cfg, epoch)
    total_loss = torch.zeros((), device=DEVICE)
    correct = torch.zeros((), device=DEVICE)
    n = 0
    for x, y in batches(data["train"], data, cfg, train=True):
        x, y_a, y_b, lam = gpu_mixup_cutmix(x, y, cfg)
        with autocast():
            logits = model(x)
            if lam < 1.0:
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total_loss += loss.detach().float() * len(y)
        target = y_a if lam >= 0.5 else y_b
        correct += (logits.argmax(1) == target).sum()
        n += len(y)
    return total_loss.item() / n, correct.item() / n


@torch.no_grad()
def evaluate(model, split, data, cfg, return_preds=False):
    model.eval()
    total_loss = torch.zeros((), device=DEVICE)
    correct = torch.zeros((), device=DEVICE)
    preds = []
    for x, y in batches(split, data, cfg, train=False):
        with autocast():
            logits = model(x)
        total_loss += F.cross_entropy(logits.float(), y, reduction="sum")
        p = logits.argmax(1)
        correct += (p == y).sum()
        if return_preds:
            preds.append(p.cpu())
    n = len(split)
    out = (total_loss.item() / n, correct.item() / n)
    return out + (torch.cat(preds).tolist(),) if return_preds else out


def fit(cfg, data, verbose=True):
    model = build_model(cfg).to(DEVICE).to(memory_format=torch.channels_last)
    opt = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP and AMP_DTYPE == torch.float16)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds")}
    best_acc, best_epoch, best_state, diverged = -1.0, 0, None, False

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        base_lr = lr_at(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = base_lr * g["lr_scale"]
        tr_loss, tr_acc = train_epoch(model, opt, scaler, criterion, data, cfg, epoch)
        if not math.isfinite(tr_loss):
            diverged = True
            print(f"  diverged at epoch {epoch + 1} (training loss {tr_loss})")
            break
        va_loss, va_acc = evaluate(model, data["val"], data, cfg)
        for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc, base_lr, time.time() - t0)):
            hist[k].append(v)
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 5 == 0 or epoch == cfg["epochs"] - 1):
            print(f"  ep {epoch + 1:3d} | lr {base_lr:.5f} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
                  f"| val loss {va_loss:.3f} acc {va_acc:.3f} | {hist['seconds'][-1]:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_epoch, diverged

# %% [markdown]
# ## Experiment runner and step-selection helpers
#
# Each step of the ladder queries the stored runs of the previous step, selects the configuration with the highest
# validation accuracy (`best_val_acc`), and applies its overrides to the next step.

# %%
def result_path(name, group, seed):
    return RESULTS_DIR / group / f"{name}_seed{seed}.json"


_last_arch = None


def run_experiment(name, group, note="", **overrides):
    global _last_arch
    cfg = {**BASE, **overrides, "name": name, "group": group}
    out = result_path(name, group, cfg["seed"])
    if out.exists() and not RETRAIN:
        run = json.loads(out.read_text())
        if run["config"] == json.loads(json.dumps(cfg)):
            print(f"skip {group}/{name}: best val {run['best_val_acc']:.3f} (already trained)")
            return run
        print(f"{group}/{name}: configuration changed since the stored run, training again")

    print(f"### {group}/{name}")
    keys = ("lr", "weight_decay", "freeze_epochs", "schedule", "layer_decay", "freeze_stages",
            "crop_scale", "random_erasing", "color_jitter", "mixup_alpha", "cutmix_alpha",
            "label_smoothing", "dropout", "seed", "epochs")
    print("    " + " | ".join(f"{k} {cfg[k]}" for k in keys))
    arch = {k: cfg[k] for k in ARCH_KEYS}
    if SHOW_SUMMARY and arch != _last_arch:
        print(show_summary(cfg))
        _last_arch = arch
    fix_random(cfg["seed"])
    data = get_data(cfg["img_size"])
    t0 = time.time()
    model, hist, best_epoch, diverged = fit(cfg, data)
    minutes = (time.time() - t0) / 60
    if best_epoch:
        test_loss, test_acc, test_pred = evaluate(model, data["test"], data, cfg, return_preds=True)
    else:
        test_loss, test_acc, test_pred = float("nan"), float("nan"), []

    cpu_model = build_model(cfg)
    run = dict(
        name=name, group=group, seed=cfg["seed"], note=note, config=cfg, history=hist,
        best_val_acc=max(hist["val_acc"], default=0.0), best_epoch=best_epoch,
        final_val_acc=hist["val_acc"][-1] if hist["val_acc"] else 0.0,\
        final_train_acc=hist["train_acc"][-1] if hist["train_acc"] else 0.0,
        test_acc=test_acc, test_loss=test_loss, test_pred=test_pred,
        test_true=data["test"].y.cpu().tolist(),
        params=count_params(cpu_model), gflops=count_gflops(cpu_model, cfg["img_size"]),
        minutes=minutes, device=torch.cuda.get_device_name() if AMP else str(DEVICE), diverged=diverged,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run))
    print(f"{name}: best val {run['best_val_acc']:.3f} (epoch {best_epoch}) | test {test_acc:.3f} | {minutes:.1f} min")
    return run


def load_group(group, results_dir=None):
    base_dir = results_dir or RESULTS_DIR
    if not (base_dir / group).exists():
        return {}
    runs = [json.loads(p.read_text()) for p in sorted((base_dir / group).glob("*.json"))]
    return {r["name"]: r for r in runs}


def run_overrides(run):
    cfg = dict(run["config"])
    if isinstance(cfg.get("crop_scale"), list):
        cfg["crop_scale"] = tuple(cfg["crop_scale"])
    return {k: v for k, v in cfg.items() if k not in ("name", "group", "seed")}


def best_in_group(group, expected_count=None):
    runs = load_group(group)
    if not runs or (expected_count is not None and len(runs) < expected_count):
        return None
    return max(runs.values(), key=lambda r: r["best_val_acc"])


def seed_mean(runs, key="best_val_acc"):
    return float(np.mean([r[key] for r in runs]))


def seed_std(runs, key="best_val_acc"):
    return float(np.std([r[key] for r in runs], ddof=1)) if len(runs) > 1 else float("nan")

# %% [markdown]
# ## Step 1: learning rate x weight decay (`s1_lr_wd`)
#
# Pretrained ImageNet weights require a smaller learning rate than random Kaiming weights so that gradient updates
# refine the existing representations rather than escaping their basin of attraction (Yosinski et al., NeurIPS 2014;
# Karpathy, CS231n). Equally important, standard $L_2$ weight decay ($\lambda \Vert{}\theta\Vert{}_2^2$) shrinks weights toward
# zero rather than toward the pretrained initialization $\theta_0$ (Li et al., *Explicit Inductive Bias for Transfer
# Learning*, ICML 2018): the heavy weight decay of Part 1 (`5e-3`) actively erases pretrained features when combined
# with a large learning rate. We search a 4 x 4 log-spaced grid of lower learning rates (`0.01` down to `0.001`) and
# weight decays (`5e-3` down to `1e-4`), keeping all other hyperparameters as in `START_2B`.

# %%
# [run]
for lr in S1_LRS:
    for wd in S1_WDS:
        run_experiment(f"s1_lr{lr:g}_wd{wd:g}", "s1_lr_wd", note="step 1: lr x wd",
                       **{**START_2B, "lr": lr, "weight_decay": wd})

s1_winner = best_in_group("s1_lr_wd", len(S1_LRS) * len(S1_WDS))
if s1_winner:
    print(f"==> Step 1 winner: {s1_winner['name']} | lr={s1_winner['config']['lr']:g}, "
          f"wd={s1_winner['config']['weight_decay']:g} | val {s1_winner['best_val_acc']:.3f} | test {s1_winner['test_acc']:.3f}")

# %% [markdown]
# ## Step 2: frozen feature extractor warmup and schedule (`s2_warmup`)
#
# At epoch 0, the 100-way linear head `fc` is initialized at random while the ResNet-18 backbone holds pretrained
# ImageNet features. Updating all layers immediately propagates large random gradients from the head into the backbone,
# distorting its representations — a phenomenon analyzed by Kumar et al. (*Fine-Tuning can Distort Pretrained Features
# and Underperform Out-of-Distribution*, ICLR 2022; see also Yannic Kilcher's ICLR 2022 walkthrough). Their **LP-FT**
# (Linear Probing then Fine-Tuning) strategy aligns the linear head first while the extractor is frozen, and then
# unfreezes the full network.
#
# In this step, fixing `(lr, weight_decay)` from Step 1, we test freezing the backbone for the first
# $K \in \{0, 3, 5, 10\}$ epochs (keeping frozen BatchNorm layers in `.eval()` mode) and compare the `onecycle`
# schedule against a standard `cosine` schedule with 3 warmup epochs.

# %%
# [run]
s1_winner = best_in_group("s1_lr_wd", len(S1_LRS) * len(S1_WDS))
if s1_winner is None:
    print("Run Step 1 first")
else:
    s1_cfg = run_overrides(s1_winner)
    for sched in S2_SCHEDULES:
        for fe in S2_FREEZE_EPOCHS:
            run_experiment(f"s2_{sched}_frz{fe}ep", "s2_warmup", note="step 2: head warmup x schedule",
                           **{**s1_cfg, "schedule": sched, "freeze_epochs": fe})
    s2_winner = best_in_group("s2_warmup", len(S2_SCHEDULES) * len(S2_FREEZE_EPOCHS))
    print(f"==> Step 2 winner: {s2_winner['name']} | schedule={s2_winner['config']['schedule']}, "
          f"freeze_epochs={s2_winner['config']['freeze_epochs']} | val {s2_winner['best_val_acc']:.3f} | test {s2_winner['test_acc']:.3f}")

# %% [markdown]
# ## Step 3: progressive learning rates and lower-layer freezing (`s3_proglr`)
#
# Convolutional networks learn a hierarchy of representations: early stages (`stem`, `layer1`) detect generic edges,
# corners and textures that transfer across visual domains, whereas deeper stages (`layer3`, `layer4`, `fc`) capture
# domain-specific object parts such as engine nacelles, wingtips and tail profiles (Yosinski et al., NeurIPS 2014).
# **Discriminative fine-tuning** (Howard & Ruder, ACL 2018; Jeremy Howard, *fast.ai* course) assigns exponentially
# smaller learning rates to earlier stages ($\text{lr}_s = \text{lr} \cdot \gamma^{5-s}$), optionally freezing the
# lowest stages altogether.
#
# On top of the best configuration from Step 2, we test `layer_decay` $\gamma \in \{1.0, 0.7, 0.5, 0.3\}$ crossed with
# `freeze_stages` $\in \{0, 1, 2\}$ (`0` = all stages trainable after warmup, `1` = `stem` permanently frozen,
# `2` = `stem` and `layer1` permanently frozen).

# %%
# [run]
s2_winner = best_in_group("s2_warmup", len(S2_SCHEDULES) * len(S2_FREEZE_EPOCHS))
if s2_winner is None:
    print("Run Step 2 first")
else:
    s2_cfg = run_overrides(s2_winner)
    for ld in S3_DECAYS:
        for fs in S3_FREEZE_STAGES:
            run_experiment(f"s3_ld{ld:g}_fs{fs}", "s3_proglr", note="step 3: progressive lr x freeze stages",
                           **{**s2_cfg, "layer_decay": ld, "freeze_stages": fs})
    s3_winner = best_in_group("s3_proglr", len(S3_DECAYS) * len(S3_FREEZE_STAGES))
    print(f"==> Step 3 winner: {s3_winner['name']} | layer_decay={s3_winner['config']['layer_decay']:g}, "
          f"freeze_stages={s3_winner['config']['freeze_stages']} | val {s3_winner['best_val_acc']:.3f} | test {s3_winner['test_acc']:.3f}")

# %% [markdown]
# ## Step 4: stronger data augmentation (`s4_aug_spatial` & `s4_aug_mix`)
#
# With only ~33 training images per aircraft variant, a pretrained network whose features are preserved fits the
# training set much faster than a network trained from scratch. In Part 1, aggressive augmentations such as CutMix or
# Mixup were left out because they made optimization too hard from random initialization in 70 epochs. Starting from
# pretrained weights, however, stronger augmentation can act as an effective regularizer against memorization:
#
# - **4A (Spatial scale & occlusion)**: we sweep minimum crop area `crop_scale` $\in \{(0.5, 1.0), (0.35, 1.0), (0.25, 1.0)\}$
#   to force multi-scale part recognition, crossed with `random_erasing` probability $p \in \{0.0, 0.25, 0.5\}$
#   (Zhong et al., AAAI 2020) so the classifier cannot rely on a single discriminative landmark.
# - **4B (Photometric jitter & regional/linear mixing)**: on top of the best 4A setup, we test fast GPU brightness/contrast
#   jitter (`color_jitter=0.2`), **Mixup** ($\alpha=0.2$, Zhang et al., ICLR 2018), and **CutMix** ($\alpha \in \{0.5, 1.0\}$,
#   Yun et al., ICCV 2019).

# %%
# [run]
s3_winner = best_in_group("s3_proglr", len(S3_DECAYS) * len(S3_FREEZE_STAGES))
if s3_winner is None:
    print("Run Step 3 first")
else:
    s3_cfg = run_overrides(s3_winner)
    # 4A: crop_scale x random_erasing
    for cs in S4_CROPS:
        for re in S4_ERASING:
            c_tag = int(round(cs[0] * 100))
            run_experiment(f"s4a_crop{c_tag}_re{re:g}", "s4_aug_spatial", note="step 4A: crop scale x random erasing",
                           **{**s3_cfg, "crop_scale": cs, "random_erasing": re})
    s4a_winner = best_in_group("s4_aug_spatial", len(S4_CROPS) * len(S4_ERASING))
    print(f"==> Step 4A winner: {s4a_winner['name']} | crop_scale={tuple(s4a_winner['config']['crop_scale'])}, "
          f"random_erasing={s4a_winner['config']['random_erasing']:g} | val {s4a_winner['best_val_acc']:.3f}")

    # 4B: color jitter, Mixup and CutMix on top of 4A winner
    s4a_cfg = run_overrides(s4a_winner)
    for mix_name, mix_over in S4_MIX_CONFIGS:
        run_experiment(f"s4b_{mix_name}", "s4_aug_mix", note="step 4B: jitter / mixup / cutmix",
                       **{**s4a_cfg, **mix_over})
    s4_winner = best_in_group("s4_aug_mix", len(S4_MIX_CONFIGS))
    print(f"==> Step 4 winner: {s4_winner['name']} | val {s4_winner['best_val_acc']:.3f} | test {s4_winner['test_acc']:.3f}")

# %% [markdown]
# ## Step 5: classifier regularization, 3-seed confirmation and 150-epoch training
#
# Finally, we recalibrate the classifier-level regularizers — **label smoothing** $\varepsilon \in \{0.0, 0.05, 0.1, 0.2\}$
# (Szegedy et al., CVPR 2016) and **dropout** $p \in \{0.0, 0.2, 0.4\}$ before `fc` (Srivastava et al., JMLR 2014) —
# since the optimal smoothing and dropout under a low-LR, strongly augmented fine-tuning regime may differ from the
# from-scratch recipe (`label_smoothing=0.2, dropout=0.0`).
#
# To guard against single-seed selection bias, we retrain the **top 2 configurations** of Step 5 with two additional
# seeds (`43, 44`) and select the final recipe by **highest mean validation accuracy over the three seeds**. We then
# train the winning recipe for **150 epochs** across all three seeds to match the 150-epoch budget of Part 1 and Part 2A.

# %%
# [run]
s4_winner = best_in_group("s4_aug_mix", len(S4_MIX_CONFIGS))
if s4_winner is None:
    print("Run Step 4 first")
else:
    s4_cfg = run_overrides(s4_winner)
    for ls in S5_SMOOTH:
        for dp in S5_DROPOUT:
            run_experiment(f"s5_ls{ls:g}_dp{dp:g}", "s5_reg", note="step 5: label smoothing x dropout",
                           **{**s4_cfg, "label_smoothing": ls, "dropout": dp})

    s5_runs = load_group("s5_reg")
    if len(s5_runs) >= len(S5_SMOOTH) * len(S5_DROPOUT):
        top_s5 = sorted(s5_runs.values(), key=lambda r: -r["best_val_acc"])[:TOP_K]
        for r in top_s5:
            over = run_overrides(r)
            for seed in SEEDS[1:]:
                run_experiment(f"{r['name']}_s{seed}", "s5_seeds", note="step 5: seed confirmation",
                               seed=seed, **over)


def confirmed_s5():
    s5_runs = load_group("s5_reg")
    if len(s5_runs) < len(S5_SMOOTH) * len(S5_DROPOUT):
        return {}
    top_s5 = sorted(s5_runs.values(), key=lambda r: -r["best_val_acc"])[:TOP_K]
    extra = load_group("s5_seeds")
    out = {}
    for r in top_s5:
        seeds_list = [r] + [extra[f"{r['name']}_s{s}"] for s in SEEDS[1:] if f"{r['name']}_s{s}" in extra]
        out[r["name"]] = seeds_list
    return out


def best_2b_recipe():
    conf = {k: v for k, v in confirmed_s5().items() if len(v) == len(SEEDS)}
    if not conf:
        return None
    best_name = max(conf, key=lambda k: seed_mean(conf[k]))
    return best_name, conf[best_name]


# [run] 150-epoch final training on 3 seeds
best_2b = best_2b_recipe()
if best_2b is not None:
    best_name, best_runs_70 = best_2b
    win_over = run_overrides(best_runs_70[0])
    print(f"==> Winning 2B recipe (3-seed mean val {seed_mean(best_runs_70):.3f}): {best_name}")
    for seed in SEEDS:
        run_experiment(f"resnet18_2B_ep{LONG_EPOCHS}_s{seed}", "final_long",
                       note=f"2B best ({best_name}), {LONG_EPOCHS} epochs",
                       seed=seed, **{**win_over, "epochs": LONG_EPOCHS})

# %% [markdown]
# ## Results
#
# All tables and plots below are computed automatically from the stored JSON logs in `results/part2B`, `results/part2A`,
# and `results/part1`.

# %%
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=COLORS), "lines.linewidth": 2,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def summary(runs):
    rows = [dict(
        run=n,
        lr=r["config"]["lr"],
        wd=r["config"]["weight_decay"],
        sched=r["config"].get("schedule", "cosine"),
        frz_ep=r["config"].get("freeze_epochs", 0),
        ld=r["config"].get("layer_decay", 1.0),
        frz_st=r["config"].get("freeze_stages", 0),
        best_val=r["best_val_acc"],
        best_epoch=r["best_epoch"],
        final_train=r["final_train_acc"],
        gap=r["final_train_acc"] - r["final_val_acc"],
        test=r["test_acc"],
        minutes=r["minutes"],
    ) for n, r in runs.items()]
    return pd.DataFrame(rows).set_index("run").style.format({
        "lr": "{:g}", "wd": "{:g}", "ld": "{:g}", "best_val": "{:.3f}",
        "final_train": "{:.3f}", "gap": "{:+.3f}", "test": "{:.3f}", "minutes": "{:.1f}"
    })


def seed_summary(groups):
    rows = [dict(
        run=label,
        epochs=runs[0]["config"]["epochs"],
        lr=runs[0]["config"]["lr"],
        wd=runs[0]["config"]["weight_decay"],
        seeds=len(runs),
        params_M=runs[0]["params"] / 1e6,
        val_mean=seed_mean(runs),
        val_std=seed_std(runs),
        test_mean=seed_mean(runs, "test_acc"),
        test_std=seed_std(runs, "test_acc"),
        train_mean=seed_mean(runs, "final_train_acc"),
    ) for label, runs in groups.items() if runs]
    return pd.DataFrame(rows).set_index("run").style.format({
        "lr": "{:g}", "wd": "{:g}", "params_M": "{:.2f}",
        "val_mean": "{:.3f}", "val_std": "{:.3f}",
        "test_mean": "{:.3f}", "test_std": "{:.3f}", "train_mean": "{:.3f}"
    })


def plot_curves(runs, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for i, (name, r) in enumerate(runs.items()):
        h = r["history"]
        ep = range(1, len(h["val_acc"]) + 1)
        c = COLORS[i % len(COLORS)]
        axes[0].plot(ep, h["val_acc"], color=c, label=name)
        if r["best_epoch"]:
            axes[0].plot(r["best_epoch"], r["best_val_acc"], "o", color=c, markersize=7)
        axes[1].plot(ep, h["val_loss"], color=c, label=name)
        axes[2].plot(ep, h["train_acc"], color=c, label=name)
    for ax, t in zip(axes, ("validation accuracy (dot = best epoch)", "validation loss (plain CE)",
                            "training accuracy")):
        ax.set_title(t); ax.set_xlabel("epoch")
    axes[0].legend(fontsize=8)
    fig.suptitle(title); plt.tight_layout(); plt.show()


def grid_table(runs, row_key, col_key, metric="best_val_acc"):
    records = [{row_key: r["config"][row_key], col_key: r["config"][col_key], metric: r[metric]}
               for r in runs.values()]
    df = pd.DataFrame(records).pivot(index=row_key, columns=col_key, values=metric)
    return df.style.format("{:.3f}").background_gradient(cmap="Blues", axis=None)

# %% [markdown]
# ### Step 1 results: learning rate x weight decay
#
# Validation accuracy across the 4 x 4 grid (`lr` x `weight_decay`), and training curves of the best weight decay at
# each learning rate.

# %%
s1_runs = load_group("s1_lr_wd")
if s1_runs:
    display(grid_table(s1_runs, "lr", "weight_decay"))
    best_per_lr = {}
    for lr in S1_LRS:
        cands = [r for r in s1_runs.values() if r["config"]["lr"] == lr]
        if cands:
            best_r = max(cands, key=lambda r: r["best_val_acc"])
            best_per_lr[f"lr={lr:g} (wd={best_r['config']['weight_decay']:g})"] = best_r
    plot_curves(best_per_lr, "Step 1: best weight decay for each learning rate")

# %% [markdown]
# ### Step 2 results: frozen feature extractor warmup and schedule

# %%
s2_runs = load_group("s2_warmup")
if s2_runs:
    display(grid_table(s2_runs, "schedule", "freeze_epochs"))
    display(summary(s2_runs))
    plot_curves(s2_runs, "Step 2: frozen feature extractor warmup epochs & schedule")

# %% [markdown]
# ### Step 3 results: progressive learning rates and stage freezing

# %%
s3_runs = load_group("s3_proglr")
if s3_runs:
    display(grid_table(s3_runs, "layer_decay", "freeze_stages"))
    best_per_ld = {}
    for ld in S3_DECAYS:
        cands = [r for r in s3_runs.values() if r["config"]["layer_decay"] == ld]
        if cands:
            best_r = max(cands, key=lambda r: r["best_val_acc"])
            best_per_ld[f"layer_decay={ld:g} (freeze={best_r['config']['freeze_stages']})"] = best_r
    plot_curves(best_per_ld, "Step 3: progressive learning-rate decay across stages")

# %% [markdown]
# ### Step 4 results: stronger data augmentation (4A spatial & 4B mixing)

# %%
s4a_runs = load_group("s4_aug_spatial")
s4b_runs = load_group("s4_aug_mix")
if s4a_runs:
    print("Step 4A: minimum crop scale x random erasing (validation accuracy)")
    records = [{"crop_scale": str(tuple(r["config"]["crop_scale"])),
                "random_erasing": r["config"]["random_erasing"],
                "best_val_acc": r["best_val_acc"]} for r in s4a_runs.values()]
    display(pd.DataFrame(records).pivot(index="crop_scale", columns="random_erasing", values="best_val_acc")
            .style.format("{:.3f}").background_gradient(cmap="Blues", axis=None))
if s4b_runs:
    print("Step 4B: color jitter, Mixup and CutMix on top of 4A winner")
    display(summary(s4b_runs))
    plot_curves(s4b_runs, "Step 4B: photometric jitter, Mixup and CutMix")

# %% [markdown]
# ### Step 5 results: classifier regularization and 3-seed confirmation

# %%
s5_runs = load_group("s5_reg")
if s5_runs:
    display(grid_table(s5_runs, "label_smoothing", "dropout"))
conf_s5 = confirmed_s5()
if conf_s5:
    display(seed_summary(conf_s5))

# %% [markdown]
# ### The Part 2B fine-tuning ladder (2A -> s1 -> s2 -> s3 -> s4 -> s5)
#
# Summary of the winning configuration at each step of the ladder (seed 42), compared against the Part 2A baseline.

# %%
p2a_runs = list(load_group("finetune", PART2A_DIR).values())
ladder_2b = {}
if p2a_runs:
    ladder_2b["2A_baseline (ep150)"] = p2a_runs[0]
for label, grp in [
    ("s1_lr_wd",     "s1_lr_wd"),
    ("s2_warmup",    "s2_warmup"),
    ("s3_proglr",    "s3_proglr"),
    ("s4_augment",   "s4_aug_mix"),
    ("s5_reg",       "s5_reg"),
]:
    w = best_in_group(grp)
    if w is not None:
        ladder_2b[label] = w

if ladder_2b:
    display(summary(ladder_2b))
    names = list(ladder_2b.keys())
    xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(xs - 0.2, [ladder_2b[n]["best_val_acc"] for n in names], 0.38, label="val (best epoch)", color=COLORS[0])
    ax.bar(xs + 0.2, [ladder_2b[n]["test_acc"] for n in names], 0.38, label="test", color=COLORS[1])
    for x_, n in zip(xs, names):
        ax.text(x_ - 0.2, ladder_2b[n]["best_val_acc"] + 0.008, f"{ladder_2b[n]['best_val_acc']:.3f}",
                ha="center", fontsize=8)
        ax.text(x_ + 0.2, ladder_2b[n]["test_acc"] + 0.008, f"{ladder_2b[n]['test_acc']:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(xs, names, rotation=15)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.70, ls="--", c="gray", lw=1, label="70% target")
    ax.legend()
    ax.set_title("Part 2B fine-tuning ladder: validation and test accuracy across steps (seed 42)")
    plt.tight_layout(); plt.show()

    plot_curves(ladder_2b, "Part 2B ladder progression (seed 42)")

# %% [markdown]
# ### Final comparison across Part 1, Part 2A and Part 2B (mean ± std over 3 seeds)
#
# We compare the three-seed results of:
# - **Part 1 from scratch** (`res_2222` and `res_3461`, 150 epochs, `results/part1/long`);
# - **Part 2A baseline fine-tuning** (`resnet18_ft`, 150 epochs with Part 1 recipe, `results/part2A/finetune`);
# - **Part 2B tuned fine-tuning** at **70 epochs** (`s5` winner, 3 seeds) and **150 epochs** (`final_long`, 3 seeds).

# %%
final_groups = {}
if PART1_DIR.exists():
    p1_long = load_group("long", PART1_DIR)
    for net_name in ("res_2222", "res_3461"):
        runs_net = [p1_long[f"{net_name}_ep150_s{s}"] for s in SEEDS if f"{net_name}_ep150_s{s}" in p1_long]
        if runs_net:
            final_groups[f"{net_name} (from scratch, Part 1, 150ep)"] = runs_net

if p2a_runs:
    final_groups["resnet18_ft (pretrained 2A, Part 1 recipe, 150ep)"] = p2a_runs

best_2b = best_2b_recipe()
if best_2b is not None:
    final_groups["resnet18_2B (pretrained 2B tuned, 70ep)"] = best_2b[1]

long_2b = [load_group("final_long")[f"resnet18_2B_ep{LONG_EPOCHS}_s{s}"]
           for s in SEEDS if f"resnet18_2B_ep{LONG_EPOCHS}_s{s}" in load_group("final_long")]
if long_2b:
    final_groups[f"resnet18_2B (pretrained 2B tuned, {LONG_EPOCHS}ep)"] = long_2b

if final_groups:
    display(seed_summary(final_groups))
    for label, runs in final_groups.items():
        print(f"{label:50s} val {seed_mean(runs):.3f} ± {seed_std(runs):.3f} | "
              f"test {seed_mean(runs, 'test_acc'):.3f} ± {seed_std(runs, 'test_acc'):.3f}")
    plot_curves({label: runs[0] for label, runs in final_groups.items()},
                "Part 1 (from scratch) vs. Part 2A (untuned FT) vs. Part 2B (tuned FT) — seed 42")
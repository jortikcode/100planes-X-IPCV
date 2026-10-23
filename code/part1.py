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
# We train a convolutional network from scratch on the 100 aircraft variants of FGVC-Aircraft, using only PyTorch
# primitives (`nn.Conv2d`, `nn.BatchNorm2d`, `nn.Linear`, ...). Our goal is a network in which every component is
# justified by a measurement, so we build it step by step and let each result motivate the next experiment:
#
# ![roadmap](attachment:roadmap.png)
#
# *Validation accuracy after each step of the study (generated from the stored results by
# `figures/make_figures.py`). The
# architecture steps add one change each; the following rows carry forward the network selected by an experiment
# or a search (grey dots: the other networks compared). Error bars: standard deviation over three seeds.*
#
# 1. **Roadmap.** We start from a plain VGG-style network and add one design choice at a time (BatchNorm, global
#    average pooling, augmentation, a fourth stage), so that every difference in accuracy has a single cause.
# 2. **Depth and residual connections.** On top of the last network of the roadmap, we compare plain and residual
#    networks of increasing depth.
# 3. **Training recipe.** The three best networks receive the same hyperparameter search over the regularizers and the
#    learning-rate schedule.
# 4. **Longer training.** The tuned networks are trained for 150 epochs instead of 70, and the final model is selected.
#
# **Protocol.** Unless stated otherwise, all runs use 224x224 images, 70 epochs, batch 64 and SGD with momentum. Each
# configuration is trained at three learning rates and represented by the best one. All choices are made on the
# validation split; the test accuracy is reported but never used to select anything.
#
# **Result.** The final model is a residual network with 31 convolutional layers, trained for 150 epochs with the tuned
# recipe: **74.0% ± 0.7 validation and 74.2% ± 0.2 test accuracy** (mean ± standard deviation over three seeds).
#
# The roadmap follows the approach of *A ConvNet for the 2020s* (Liu et al., CVPR 2022), which modernizes a ResNet one
# design choice at a time: a cumulative ablation, in which each step keeps all the previous ones.
#
# *Lectures are those of part 2 of the course Image Processing and Computer Vision, cited by slide title.*

# %% [markdown]
# ## Setup
#
# The notebook runs top to bottom on Colab (GPU runtime) or locally. Every run reported below is stored in the
# repository (`results/part1`) and is loaded instead of being trained again, so the whole notebook reproduces our
# tables and plots in a few minutes. Flags in the next cell:
#
# - `RETRAIN = True` trains every run again (about 3 hours on an A100);
# - `USE_DRIVE` (Colab only) keeps results and the image cache in Google Drive across sessions;
# - `VERBOSE` prints one line per stored run instead of a summary, `SHOW_SUMMARY` prints the architecture of every new
#   run.

# %% id="Me_GVIWbQLJU"
# %pip install -q torchinfo kagglehub

# %% colab={"base_uri": "https://localhost:8080/"} id="d05ZbaRIQLJV" outputId="f8a4856d-45ab-4a9f-9284-4daca6ebd263"
import contextlib, json, math, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from torchvision import datasets
from torchvision.ops import roi_align
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

IN_COLAB = "google.colab" in sys.modules
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

# Mixed precision on CUDA only: bfloat16 where supported (A100, L4), float16 + loss scaling otherwise (T4)
AMP = DEVICE.type == "cuda"
AMP_DTYPE = torch.bfloat16 if (AMP and torch.cuda.is_bf16_supported()) else torch.float16

RETRAIN = False        # True: retrain runs even if their JSON already exists
SHOW_SUMMARY = False   # print the model summary when a run starts with a different architecture than the last one
VERBOSE = False        # print one line for every stored run that is loaded instead of trained
USE_DRIVE = False       # Colab only: keep results and image cache in Google Drive (False: Colab's temporary disk)
print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name()})" if AMP else "")
      + f" | amp: {AMP_DTYPE if AMP else 'off'} | torch {torch.__version__}")

# %% [markdown]
# ### Paths
#
# On Colab, results and the preprocessed images live in Google Drive when `USE_DRIVE = True`, on the temporary disk of
# the session otherwise; in both cases the stored results of the repository are downloaded first. Locally, they are read
# from the repository itself.

# %% colab={"base_uri": "https://localhost:8080/"} id="QE6sNXgIQLJW" outputId="0bae8747-f4d5-427f-9658-5d740e12273c"
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS_DIR = Path("/content/drive/MyDrive/ipcv/results/part1")
    CACHE_DIR = Path("/content/drive/MyDrive/ipcv/cache")
    DATA_ROOT = Path("/content/data")
elif IN_COLAB:
    # everything on the temporary disk of the session: no access to Google Drive is needed
    RESULTS_DIR = Path("/content/results/part1")
    CACHE_DIR = Path("/content/cache")
    DATA_ROOT = Path("/content/data")
else:
    REPO_DIR = Path.cwd().parent if Path.cwd().name == "code" else Path.cwd()
    RESULTS_DIR = REPO_DIR / "results" / "part1"
    DATA_ROOT = Path.home() / "data" / "fgvc"
    CACHE_DIR = DATA_ROOT / "cache"

for d in (RESULTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
print("results ->", RESULTS_DIR)

GITHUB_REPO, GITHUB_REF = "jortikcode/100planes-X-IPCV", "main"   # a branch, or a tag to freeze the results
FETCH_RESULTS = IN_COLAB      # locally the repo already contains them


def fetch_results(repo, ref, dest):
    # Download the repo as one archive and copy its results/part1 JSON files into dest, never overwriting.
    import io, tarfile, urllib.request
    with urllib.request.urlopen(f"https://github.com/{repo}/archive/{ref}.tar.gz") as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
    added = kept = 0
    for member in archive.getmembers():
        path = member.name.split("/", 1)[-1]                 # drop the "<repo>-<ref>/" prefix
        name = path.rsplit("/", 1)[-1]
        if (not (member.isfile() and path.startswith("results/part1/") and name.endswith(".json"))
                or ".." in path or name.startswith(".")):
            continue
        target = dest / path[len("results/part1/"):]
        if target.exists():
            kept += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.extractfile(member).read())
        added += 1
    print(f"results from github.com/{repo} ({ref}): {added} downloaded, {kept} already present")


if FETCH_RESULTS:
    fetch_results(GITHUB_REPO, GITHUB_REF, RESULTS_DIR)


# %% [markdown]
# ### Reproducibility
#
# Every source of randomness is seeded (weight initialization, data order, augmentation) and cuDNN runs in deterministic
# mode: on the same hardware and software, a configuration trained twice with the same seed gives identical curves
# and predictions (verified at the end of the notebook). Training the same configuration with different seeds is how we
# measure run-to-run noise.

# %% id="2ZeSxwHNQLJW"
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
# A base configuration holds every choice shared by all runs; each experiment changes only what it studies.
#
# - **Optimizer: SGD with momentum 0.9** — the optimizer of VGG and ResNet (lecture 3, *SGD with minibatches*; lecture 5).
# - **Learning-rate sweep.** Each configuration is trained at learning rates 0.1, 0.05 and 0.01 (log-spaced, lecture 6 *Random
#   hyper-parameters search*) and represented by its best one. The best learning rate depends on the network — without
#   BatchNorm, or with a large classification head, training becomes unstable at values that a normalized network
#   handles well — so a single value for every step could make a step look harmful only because the value does not
#   suit it.
# - **Schedule: linear warmup for 3 epochs, then cosine decay to zero** (lecture 6, *Learning rate schedule: cosine* and
#   *warm-up*). A large learning rate makes fast progress but cannot settle into a minimum; decaying it lets the
#   network settle, and starting low avoids large updates while the weights are still random.
# - **Budget: 70 epochs, batch 64** — a compromise between convergence and the number of runs the study requires; the
#   selected networks are trained longer at the end.
# - **Resolution: 224x224**, the standard resolution of ImageNet training and of the pretrained ResNet-18 of Part 2.
# - **Regularizers off** (weight decay, label smoothing, dropout, random erasing) until the training recipe, so that the
#   roadmap and the depth experiment measure the architecture alone.

# %% colab={"base_uri": "https://localhost:8080/"} id="xKM_uuC8QLJW" outputId="3c3fa92b-330c-491b-a4ea-53df25e59ff4"
BASE = dict(
    seed=42,
    # data
    img_size=224,
    augment=False,
    crop_scale=(0.5, 1.0),           # RandomResizedCrop: area of the crop, as a fraction of the image
    # architecture
    stem=(32, 32, 64),
    channels=(64, 128, 256),
    blocks=(1, 1, 1),                # blocks per stage; a block is two 3x3 convs
    batchnorm=False,
    gap=False,
    residual=False,
    dropout=0.0,                     # before the classifier; searched in the training recipe
    # training
    epochs=70,
    batch_size=64,
    lr=0.05,                         # overridden by the sweep
    momentum=0.9,
    weight_decay=0.0,                # searched in the training recipe
    label_smoothing=0.0,             # searched in the training recipe
    random_erasing=0.0,              # probability of erasing a random rectangle; searched in the training recipe
    warmup_epochs=3,
)

# every configuration is trained at each of these learning rates and represented by its best one (on validation)
LR_SWEEP = [0.1, 0.05, 0.01]

# extra seeds for the depth x residual networks and for the recipe (seed 42 comes from the sweep)
SEEDS = [42, 43, 44]

# The ladder: each step adds exactly one change on top of all the previous ones
STEPS = [
    ("s0_vgg",    dict()),
    ("s1_bn",     dict(batchnorm=True)),
    ("s2_gap",    dict(gap=True)),
    ("s3_aug",    dict(augment=True)),
    ("s4_stage4", dict(channels=(64, 128, 256, 512), blocks=(1, 1, 1, 1))),
]


def cumulative(steps):
    # [(name, overrides of that step and of all the steps before it)]
    acc, out = {}, []
    for name, change in steps:
        acc = {**acc, **change}
        out.append((name, dict(acc)))
    return out


LADDER = cumulative(STEPS)
S4 = LADDER[-1][1]

# Depth x residual, on top of s4: three depths, each with and without skip connections (see the section for why)
GRID = [
    ("plain_1111", dict(S4, blocks=(1, 1, 1, 1), residual=False)),   # = s4
    ("res_1111",   dict(S4, blocks=(1, 1, 1, 1), residual=True)),
    ("plain_2222", dict(S4, blocks=(2, 2, 2, 2), residual=False)),   # ResNet-18 layout
    ("res_2222",   dict(S4, blocks=(2, 2, 2, 2), residual=True)),
    ("plain_3461", dict(S4, blocks=(3, 4, 6, 1), residual=False)),   # ResNet-34 layout, one 512-block
    ("res_3461",   dict(S4, blocks=(3, 4, 6, 1), residual=True)),
]

# The regularizers of the training recipe, at the values they keep during step 1 of its search (lr x weight decay)
RECIPE = dict(weight_decay=5e-4, label_smoothing=0.1, dropout=0.2, random_erasing=0.25)

# Recipe tuning (see its section): the TUNE_N best networks of the grid get the same search
TUNE_N = 3
TUNE_LRS = [0.1, 0.05]                      # grid search: learning rate x weight decay (log-spaced)
TUNE_WDS = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
SEARCH_SPACE = dict(                        # random search over the other recipe parameters
    label_smoothing=[0.0, 0.05, 0.1, 0.2],
    dropout=[0.0, 0.2, 0.5],
    random_erasing=[0.0, 0.25, 0.5],
    crop_scale=[(0.25, 1.0), (0.35, 1.0), (0.5, 1.0)],
    schedule=["cosine", "onecycle"],
)
N_RANDOM = 12            # random configurations, the same list for every network
SEARCH_SEED = 0          # seed of the random draw of the configurations (not of the trainings)
TOP_K = 2                # best configurations of each network confirmed with the other seeds

# Longer training: the tuned networks, each with its best configuration, trained for more epochs
LONG_EPOCHS = 150

# %% [markdown]
# ## Data
#
# FGVC-Aircraft (`torchvision.datasets.FGVCAircraft`, `annotation_level="variant"`) has 100 classes and 3,334 /
# 3,333 / 3,333 images in the training / validation / test splits: about **33 training images per class**. With so few images a network can
# memorize the training set long before it learns features that generalize, and this is the problem most of the
# following steps address.
#
# ![data pipeline](attachment:data_pipeline.png)
#
# **Preprocessing, once per split** (top row). We crop the 20-pixel copyright banner at the bottom of every image, so that the
# network cannot rely on it, and resize the image to 224x224 *without* keeping the aspect ratio. The usual alternative
# — resize the short side and crop the centre — would cut the left and right edges, and in this dataset the aircraft
# spans almost the whole width of the image: a centre crop removes the nose and the tail, where variants of the same
# family differ. Since almost all images share the same landscape aspect ratio, the stretch is the same fixed squeeze for
# every image. Normalization uses the per-channel mean and standard deviation of the training split only.
#
# **Augmentation, on training batches only** (bottom row; lecture 6, *Data augmentation*). Random resized crops (50–100% of the area) and
# horizontal flips change scale and position while preserving the label — a mirrored aircraft is still the same
# aircraft; random erasing is added by the training recipe. We exclude vertical flips and large rotations, which produce
# images that never occur at test time, and colour jitter, since colour mostly encodes the airline livery rather than
# the variant.
#
# **Efficiency.** Decoding and resizing the full-resolution JPEGs at every epoch would make the CPU the bottleneck, so
# each split is preprocessed once and cached as a tensor of 8-bit integers. During training the whole split is kept on the GPU,
# and the augmentation is computed per batch on the GPU: the 64 random crops of a batch are extracted and resized by a
# single `roi_align` call, which gives the same result as `RandomResizedCrop`. The dataset is always loaded through the
# torchvision class; when its official server is unavailable, the identical `fgvc-aircraft-2013b` archive is downloaded
# from a public Kaggle mirror and placed where the class expects it.

# %% id="cQ29ylTcQLJX"
BANNER_PX = 20
KAGGLE_MIRROR = "seryouxblaster764/fgvc-aircraft"     # public copy of the official fgvc-aircraft-2013b archive


def fetch_from_kaggle(root):
    # Download the official archive from the Kaggle mirror and place it where torchvision's FGVCAircraft expects it.
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
    # an alternative file name for the same cache (same preprocessing: banner crop, stretch to a square)
    legacy = CACHE_DIR / f"{split}_{img_size}.pt"
    for p in (path, legacy):
        if p.exists():
            return torch.load(p)
    try:
        dset = datasets.FGVCAircraft(DATA_ROOT, split=split, annotation_level="variant", download=True)
    except Exception as official_error:
        # the official server is sometimes unavailable: same archive from the Kaggle mirror, still read by torchvision
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
    # One split held on the device as uint8 images + labels.
    def __init__(self, split, img_size):
        x, y, self.classes = build_cache(split, img_size)
        self.x, self.y = x.to(DEVICE), y.to(DEVICE)

    def __len__(self):
        return len(self.y)


_DATA = {}


def get_data(img_size):
    # The three splits at one resolution, plus normalization statistics measured on train only.
    if img_size not in _DATA:
        splits = {s: GPUSplit(s, img_size) for s in ("train", "val", "test")}
        x = splits["train"].x.float() / 255
        mean = x.mean(dim=(0, 2, 3)).view(1, 3, 1, 1)
        std = x.std(dim=(0, 2, 3)).view(1, 3, 1, 1)
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
    boxes = torch.stack([idx, x0, y0, x0 + w, y0 + h], dim=1)      # one box per image: (index, x1, y1, x2, y2)
    out = roi_align(x, boxes, output_size=size, aligned=True)      # crop every box and resize it to size x size
    flip = torch.rand(B, device=dev) < 0.5
    return torch.where(flip.view(B, 1, 1, 1), out.flip(-1), out)          # flip half of the images


def gpu_random_erasing(x, p, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
    # RandomErasing on a normalized batch: with probability p, a random rectangle of each image is set to 0 (the mean).
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


def batches(split, data, cfg, train):
    # Yields normalized (x, y) batches; train=True shuffles, drops the last partial batch and augments.
    n = len(split)
    order = torch.randperm(n).to(DEVICE) if train else torch.arange(n, device=DEVICE)
    bs = cfg["batch_size"] if train else 512
    for i in range(0, n, bs):
        idx = order[i:i + bs]
        if train and len(idx) < bs:
            break
        x = split.x[idx].float() / 255                                       # float batch, in [0, 1]
        if train and cfg["augment"]:
            x = gpu_augment(x, cfg["img_size"], scale=tuple(cfg["crop_scale"]))
        x = (x - data["mean"]) / data["std"]                                 # normalized batch
        if train and cfg["random_erasing"] > 0:
            x = gpu_random_erasing(x, cfg["random_erasing"])
        yield x.contiguous(memory_format=torch.channels_last), split.y[idx]


# %% colab={"base_uri": "https://localhost:8080/", "height": 627, "referenced_widgets": ["56021d40a2a24aeb8e4ce236e7148752", "6bfaa424ab0c4850a6178b86192df164", "56d0fbce688e46be90203013770b34f0", "8fd45dc7b2f34c8b991c5e0c6372b909", "384e443d54af484388c872cd9c651db6", "a5ecdadd79b54812b504fbdb18a9e77a", "02254a46ef3a4539b3e88e2c76de0b54", "3bdf4bc01bd04181857632f348b260b2", "f83a31387aff4eff9ef9ca5018fb8480", "876bf5b0cca245798b08a3e1766fa7b0", "9523291372d048e4bdad61970d4317c5", "d00deb33dced4c739082b3f66fe2f28a", "54644bff45634c01ba9e198d92dda2a3", "c55a9f91ffa4417ca5953e7980538c54", "235f4ce923b34c8fa7a67eb53c9ce437", "f3aa4a0d36b4401dba9f15d53928dd12", "a7be3d6e83bd43c9ab94d446a3a84092", "3be9ba871a9b47a6966b9a407541e07f", "44a0209d1dd24772bba80aaec434ec2b", "24a8fb733b6149f8bed15a6f62f533a3", "63168f7aedf14b8baf77319c63b9e1e8", "f9ccf2160eb34d3a83c04c8123d81006", "18a3bab19b14462b94c01db5d92d4e5a", "3d89a74f9dd24957a44885ee6b678ab7", "7c651a7c0668478584b8a36e77df11e5", "0e5caa1c119d4c25a340b93b9a273426", "842d760ac3fc4f4b88ea9810b87842b3", "ff48c049053b4ab6a5be12897136ee39", "e9c5b8b8a0af4a5c94c7c9921a9f794f", "9dc002e952fa4947b0cc0d5f3fcaefeb", "4ce45873e5b448128bc1e50bcf8c2736", "5809a734924c49dca6fec8b08bcfea63", "08f8bcd48a7f4c799c71e80e8ca1709e"]} id="Mt7nFLOrQLJX" outputId="b731a5fa-81ae-4e7a-f570-21b36df716db"
try:
    data = get_data(BASE["img_size"])
except Exception as e:
    data = None
    print(f"WARNING: the dataset is not available ({e})\n"
          "The cells that need the images are skipped; the stored results are still loaded and shown.")

N_CLASSES = 100                                   # FGVC-Aircraft, annotation_level="variant"
if data is not None:
    CLASSES = data["train"].classes
    assert len(CLASSES) == N_CLASSES
    print(f"classes: {N_CLASSES} | train {len(data['train'])} | val {len(data['val'])} | test {len(data['test'])}")


# %% [markdown]
# ## The network
#
# A single network class builds every network of the study from its configuration. In the figure, the components
# added by a step of the roadmap have the colour of that step in the roadmap chart; $n_1, \dots, n_4$ are the
# numbers of blocks per stage, varied in section 2:
#
# ![network](attachment:network.png)
#
# - **Stages** (lecture 5, *VGG: deep but regular*, *Stages*). The network is a stem followed by stages of identical blocks;
#   every stage after the first halves the resolution and doubles the channels. A block is two 3x3 convolutions, which
#   cover the receptive field of a 5x5 convolution with fewer parameters and one more non-linearity.
# - **Stem** (lecture 5, *Stem layers*; *Further model tweaks — ResNet-C*). The resolution is reduced by 4 before the first
#   stage, as in ResNet and GoogLeNet, since early layers detect simple patterns that need no full resolution and the
#   blocks are 16 times cheaper at 56x56 than at 224x224. As in ResNet-C, the 7x7 convolution of ResNet is replaced by
#   three 3x3 convolutions (32, 32, 64 channels), the first one with stride 2.
# - **BatchNorm** (lecture 4, *Batch Norm*; Ioffe & Szegedy, 2015) normalizes the activations of each channel over the batch.
#   It allows higher learning rates and makes initialization less critical. It is placed after every convolution,
#   before the ReLU, and the convolutions it follows have no bias.
# - **Global average pooling** (lecture 5, *Fully-connected classifier vs global average pooling*; Lin et al., 2014) averages
#   each channel of the last feature map, replacing a flatten layer whose classifier would hold most of the parameters
#   of the network.
# - **Residual connections** (lecture 5, *Residual Networks*, *Residual block*, *Skip conn dimensions*; He et al., 2016).
#   The block computes $F(x) + x$: it only has to learn a correction of the identity, and very deep networks become
#   trainable. When a block changes resolution or channels, the input goes through a 1x1 convolution with BatchNorm
#   (projection); otherwise it is added unchanged. Plain networks do not create the projections, so plain and residual
#   versions of a network have almost the same number of parameters.
# - **Initialization:** Kaiming normal (He et al., 2015), which keeps the scale of the activations stable through depth.

# %% id="HlGeLbfvQLJX"
def conv_bn(c_in, c_out, k, stride, bn):
    layers = [nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Block(nn.Module):
    def __init__(self, c_in, c_out, stride, bn, residual):
        super().__init__()
        self.conv1 = conv_bn(c_in, c_out, 3, stride, bn)
        self.conv2 = conv_bn(c_out, c_out, 3, 1, bn)
        self.residual = residual
        if residual and (stride != 1 or c_in != c_out):
            self.skip = conv_bn(c_in, c_out, 1, stride, bn)      # projection
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.conv2(F.relu(self.conv1(x)))
        if self.residual:
            out = out + self.skip(x)
        return F.relu(out)


class AircraftNet(nn.Module):
    def __init__(self, cfg, n_classes=100):
        super().__init__()
        bn = cfg["batchnorm"]
        c1, c2, c3 = cfg["stem"]
        self.stem = nn.Sequential(
            conv_bn(3, c1, 3, 2, bn), nn.ReLU(inplace=True),
            conv_bn(c1, c2, 3, 1, bn), nn.ReLU(inplace=True),
            conv_bn(c2, c3, 3, 1, bn), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        layers, c_in = [], c3
        for stage, (c_out, n_blocks) in enumerate(zip(cfg["channels"], cfg["blocks"])):
            for b in range(n_blocks):
                stride = 2 if (b == 0 and stage > 0) else 1
                layers.append(Block(c_in, c_out, stride, bn, cfg["residual"]))
                c_in = c_out
        self.stages = nn.Sequential(*layers)

        if cfg["gap"]:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Dropout(cfg["dropout"]), nn.Linear(c_in, n_classes))
        else:
            with torch.no_grad():   # size of the last feature map, for the flatten head
                s = self.stages(self.stem(torch.zeros(1, 3, cfg["img_size"], cfg["img_size"]))).shape
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(cfg["dropout"]),
                                      nn.Linear(s[1] * s[2] * s[3], n_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_gflops(model, img_size):
    # Multiply-adds of convolutions and linear layers for one image, x2 = FLOPs.
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


# %% [markdown]
# Layer-by-layer summary of one network (`torchinfo`).

# %% colab={"base_uri": "https://localhost:8080/"} id="ytkI2WwNQLJX" outputId="5e73c057-45fd-47ad-a98f-7eabd781b8e2"
# the keys that define the architecture: two runs with the same values build the same network
ARCH_KEYS = ("img_size", "stem", "channels", "blocks", "batchnorm", "gap", "residual", "dropout")


def show_summary(cfg, depth=4):
    return torchinfo.summary(AircraftNet(cfg, N_CLASSES), input_size=(1, 3, cfg["img_size"], cfg["img_size"]),
                             col_names=("output_size", "num_params"), row_settings=("var_names",),
                             depth=depth, device="cpu", verbose=0)


show_summary({**BASE, **dict(GRID)["res_2222"]})

# %% [markdown]
# Size and cost of every network of the study. With the flatten classifier (the plain VGG and the network with
# BatchNorm), the classifier alone holds 81% of the parameters.

# %% colab={"base_uri": "https://localhost:8080/", "height": 425} id="tPnqfghGQLJY" outputId="abc9b23e-c89c-4f45-f4a8-e428a9d66fd0"
rows = []
for name, over in LADDER + GRID:
    cfg = {**BASE, **over}
    m = AircraftNet(cfg, N_CLASSES)
    head = sum(p.numel() for p in m.head.parameters())
    rows.append(dict(run=name, params_M=count_params(m) / 1e6, head_share=head / count_params(m),
                     GFLOPs=count_gflops(m, cfg["img_size"]), convs=sum(isinstance(l, nn.Conv2d) for l in m.modules())))
pd.DataFrame(rows).set_index("run").style.format({"params_M": "{:.2f}", "head_share": "{:.0%}", "GFLOPs": "{:.2f}"})


# %% [markdown]
# ## Training
#
# - **Loss:** cross-entropy (lecture 3, *Softmax function*, *Cross-entropy loss*), with label smoothing where the recipe enables
#   it. The validation loss is always the plain cross-entropy, so that all curves share the same scale.
# - **Weight decay**, when enabled, applies to convolution and linear weights only, not to BatchNorm parameters and
#   biases.
# - **Model selection** (lecture 6, *Early stopping*): every run keeps the epoch with the highest validation accuracy, and the
#   test split is evaluated once, on that checkpoint.
# - **Mixed precision** on CUDA (bfloat16 on A100/lecture 4, float16 with loss scaling on T4).

# %% id="XDaVeJmjQLJY"
def autocast():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if AMP else contextlib.nullcontext()


def make_optimizer(model, cfg):
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.SGD([{"params": decay, "weight_decay": cfg["weight_decay"]},
                            {"params": no_decay, "weight_decay": 0.0}],
                           lr=cfg["lr"], momentum=cfg["momentum"])


def lr_at(epoch, cfg):
    if cfg.get("schedule", "cosine") == "onecycle":
        # one-cycle (Smith & Topin, 2019), per epoch: cosine rise from lr/25 to lr over the first 30% of the
        # epochs, then cosine decay to lr/1e4 (the defaults of torch.optim.lr_scheduler.OneCycleLR)
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


def train_epoch(model, opt, scaler, criterion, data, cfg):
    model.train()
    total_loss = torch.zeros((), device=DEVICE)
    correct = torch.zeros((), device=DEVICE)
    n = 0
    for x, y in batches(data["train"], data, cfg, train=True):
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total_loss += loss.detach().float() * len(y)
        correct += (logits.argmax(1) == y).sum()
        n += len(y)
    return total_loss.item() / n, correct.item() / n


@torch.no_grad()
def evaluate(model, split, data, cfg, return_preds=False):
    model.eval()
    total_loss, correct, preds = 0.0, 0, []
    for x, y in batches(split, data, cfg, train=False):
        with autocast():
            logits = model(x)
        total_loss += F.cross_entropy(logits.float(), y, reduction="sum").item()
        p = logits.argmax(1)
        correct += (p == y).sum().item()
        if return_preds:
            preds.append(p.cpu())
    n = len(split)
    out = (total_loss / n, correct / n)
    return out + (torch.cat(preds).tolist(),) if return_preds else out


def fit(cfg, data, verbose=True):
    model = AircraftNet(cfg, N_CLASSES).to(DEVICE).to(memory_format=torch.channels_last)
    opt = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP and AMP_DTYPE == torch.float16)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds")}
    best_acc, best_epoch, best_state, diverged = -1.0, 0, None, False

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        lr = lr_at(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        tr_loss, tr_acc = train_epoch(model, opt, scaler, criterion, data, cfg)
        if not math.isfinite(tr_loss):
            diverged = True
            print(f"  diverged at epoch {epoch + 1} (training loss {tr_loss})")
            break
        va_loss, va_acc = evaluate(model, data["val"], data, cfg)
        for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc, lr, time.time() - t0)):
            hist[k].append(v)
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 5 == 0 or epoch == cfg["epochs"] - 1):
            print(f"  ep {epoch + 1:3d} | lr {lr:.4f} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
                  f"| val loss {va_loss:.3f} acc {va_acc:.3f} | {hist['seconds'][-1]:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_epoch, diverged


# %% [markdown]
# ### Sanity checks
#
# Before trusting any experiment we verify the training pipeline (lecture 6, *A "recipe" for training NNs*, step 2: "check
# init loss, overfit a small dataset"). At initialization the loss must be close to ln(100) ≈ 4.61, the loss of a
# uniform guess over 100 classes; and the network must be able to memorize 100 training images. A failure would point
# to a bug in the data, the labels or the training step.

# %% colab={"base_uri": "https://localhost:8080/"} id="xYWd8O-1QLJY" outputId="777bb68d-6214-4718-d7e3-45c147e04379"
if data is None:
    print("skipped: the dataset is not available")
else:
    fix_random(0)
    cfg = {**BASE, "batchnorm": True}
    model = AircraftNet(cfg, N_CLASSES).to(DEVICE)
    x, y = next(batches(data["train"], data, cfg, train=True))
    with torch.no_grad():
        print(f"initial loss {F.cross_entropy(model(x), y).item():.3f}  (expected ~{math.log(N_CLASSES):.2f})")

    # memorize 100 images with Adam, full batch
    idx = torch.arange(100, device=DEVICE)
    x = ((data["train"].x[idx].float() / 255 - data["mean"]) / data["std"])
    y = data["train"].y[idx]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(150):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        print(f"tiny subset: train acc {(model(x).argmax(1) == y).float().mean().item():.2f}  (expected ~1.00)")


# %% [markdown]
# ## Experiment runner
#
# The runner trains one configuration, evaluates its best checkpoint on the test split and stores the configuration,
# the per-epoch curves and the test predictions; a run whose result already exists with the same configuration is
# loaded instead of trained. Every configuration is first trained at the three learning rates (seed 42), then with two
# more seeds at its best learning rate when the experiment requires it. The seed-42 result is the best of three learning
# rates, hence slightly optimistic; the two other seeds are not. The plotting helpers used in the rest of the notebook
# follow.

# %% id="F0jmIWZtQLJY"
def result_path(name, group, seed):
    return RESULTS_DIR / group / f"{name}_seed{seed}.json"


_last_arch = None
RUN_STATS = {"loaded": 0, "trained": 0}


def run_report():
    # One line per training cell: how many runs were loaded from the stored results and how many were trained.
    print(f"runs loaded from the stored results: {RUN_STATS['loaded']} | trained: {RUN_STATS['trained']}")
    RUN_STATS.update(loaded=0, trained=0)


def run_experiment(name, group, note="", **overrides):
    global _last_arch
    cfg = {**BASE, **overrides, "name": name, "group": group}
    out = result_path(name, group, cfg["seed"])
    if out.exists() and not RETRAIN:
        run = json.loads(out.read_text())
        if run["config"] == json.loads(json.dumps(cfg)):      # same configuration (tuples stored as lists)
            RUN_STATS["loaded"] += 1
            if VERBOSE:
                print(f"skip {group}/{name}: best val {run['best_val_acc']:.3f} (already trained)")
            return run
        print(f"{group}/{name}: configuration changed since the stored run, training again")

    print(f"### {group}/{name}")
    print("    " + " | ".join(f"{k} {cfg[k]}" for k in ("lr", "weight_decay", "label_smoothing", "dropout",
                                                        "random_erasing", "augment", "crop_scale", "seed", "epochs"))
          + f" | schedule {cfg.get('schedule', 'cosine')}")
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
    else:   # diverged before the first validation: there is no checkpoint to test
        test_loss, test_acc, test_pred = float("nan"), float("nan"), []

    cpu_model = AircraftNet(cfg, N_CLASSES)
    run = dict(
        name=name, group=group, seed=cfg["seed"], note=note, config=cfg, history=hist,
        best_val_acc=max(hist["val_acc"], default=0.0), best_epoch=best_epoch,
        final_val_acc=hist["val_acc"][-1] if hist["val_acc"] else 0.0,
        final_train_acc=hist["train_acc"][-1] if hist["train_acc"] else 0.0,
        test_acc=test_acc, test_loss=test_loss, test_pred=test_pred,
        test_true=data["test"].y.cpu().tolist(),
        params=count_params(cpu_model), gflops=count_gflops(cpu_model, cfg["img_size"]),
        minutes=minutes, device=torch.cuda.get_device_name() if AMP else str(DEVICE), diverged=diverged,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run))
    RUN_STATS["trained"] += 1
    print(f"{name}: best val {run['best_val_acc']:.3f} (epoch {best_epoch}) | test {test_acc:.3f} | {minutes:.1f} min")
    return run


def load_group(group):
    runs = [json.loads(p.read_text()) for p in sorted((RESULTS_DIR / group).glob("*.json"))]
    return {r["name"]: r for r in runs}


def sweep_runs(runs, name):
    # The seed-42 runs of one configuration of the sweep, keyed by learning rate.
    return {r["config"]["lr"]: r for n, r in runs.items()
            if n.startswith(f"{name}_lr") and r["config"]["lr"] in LR_SWEEP and r["seed"] == SEEDS[0]}


def best_of(runs, name):
    # The best run (on validation) of one configuration of the sweep; None until every learning rate is trained.
    swept = sweep_runs(runs, name)
    if len(swept) < len(LR_SWEEP):
        return None
    return max(swept.values(), key=lambda r: r["best_val_acc"])


def run_sweep(name, group, note="", **overrides):
    for lr in LR_SWEEP:
        run_experiment(f"{name}_lr{lr:g}", group, note=note, **{**overrides, "lr": lr})
    best = best_of(load_group(group), name)
    print(f"==> {label(name)}: best learning rate {best['config']['lr']:g} | validation {best['best_val_acc']:.3f} | "
          f"test {best['test_acc']:.3f}\n")
    return best


def run_seeds(name, group, note="", **overrides):
    # The other seeds, at the learning rate that won the sweep of this configuration.
    lr = best_of(load_group(group), name)["config"]["lr"]
    for seed in SEEDS[1:]:
        run_experiment(f"{name}_lr{lr:g}_s{seed}", f"{group}_seeds", note=note, seed=seed, **{**overrides, "lr": lr})


def seed_runs(name, group):
    # Every seed of one configuration at its best learning rate: [seed 42 (from the sweep), seed 43, seed 44].
    best = best_of(load_group(group), name)
    if best is None:
        return []
    lr, others = best["config"]["lr"], load_group(f"{group}_seeds")
    return [best] + [others[f"{name}_lr{lr:g}_s{s}"] for s in SEEDS[1:] if f"{name}_lr{lr:g}_s{s}" in others]


def seed_mean(runs, key="best_val_acc"):
    return float(np.mean([r[key] for r in runs]))


def seed_std(runs, key="best_val_acc"):
    return float(np.std([r[key] for r in runs], ddof=1)) if len(runs) > 1 else float("nan")


def finalists(n=TUNE_N):
    # The n networks of the grid with the highest mean validation accuracy over the seeds.
    done = {name: seed_runs(name, "depth_residual") for name, _ in GRID}
    if any(len(r) < len(SEEDS) for r in done.values()):
        return None
    return sorted(done, key=lambda name: -seed_mean(done[name]))[:n]


def simplicity(name):
    # Sort key: fewer blocks first, then plain before residual.
    over = dict(GRID)[name]
    return (sum(over["blocks"]), over["residual"])


def random_configs():
    # The N_RANDOM configurations of the random search: drawn once, identical for every network.
    rng = np.random.RandomState(SEARCH_SEED)
    configs, seen = [], set()
    while len(configs) < N_RANDOM:
        cfg = {k: v[rng.randint(len(v))] for k, v in SEARCH_SPACE.items()}
        key = json.dumps(cfg)
        if key not in seen:
            seen.add(key)
            if cfg["schedule"] == "cosine":
                del cfg["schedule"]            # cosine is the default: keep the stored config minimal
            configs.append(cfg)
    return configs


def run_overrides(run):
    # The overrides that reproduce a stored run (everything but its name, group and seed).
    return {k: v for k, v in run["config"].items() if k not in ("name", "group", "seed")}


def tuning_runs(name):
    # Every seed-42 run of the recipe search (grid + random) of one network, keyed by run name.
    runs = {**load_group("tune_grid"), **load_group("tune_random")}
    return {n: r for n, r in runs.items() if n.startswith(name + "_")}


def best_grid_point(name):
    grid = {n: r for n, r in load_group("tune_grid").items() if n.startswith(name + "_")}
    if len(grid) < len(TUNE_LRS) * len(TUNE_WDS):
        return None
    best = max(grid.values(), key=lambda r: r["best_val_acc"])
    return best["config"]["lr"], best["config"]["weight_decay"]


def top_configs(name):
    # The TOP_K best runs of the search of one network (seed 42).
    runs = tuning_runs(name)
    return sorted(runs.values(), key=lambda r: -r["best_val_acc"])[:TOP_K]


def confirmed(name):
    # {run name: [seed 42, 43, 44]} for the TOP_K configurations of one network.
    seeds = load_group("tune_seeds")
    return {r["name"]: [r] + [seeds[f"{r['name']}_s{s}"] for s in SEEDS[1:] if f"{r['name']}_s{s}" in seeds]
            for r in top_configs(name)}


def best_tuned(name):
    # (configuration name, [runs over the seeds]) of the best configuration of the recipe search of one network.
    conf = {n: runs for n, runs in confirmed(name).items() if len(runs) == len(SEEDS)}
    if not conf:
        return None
    best = max(conf, key=lambda n: seed_mean(conf[n]))
    return best, conf[best]


def long_runs(name):
    # The longer trainings of one network, over the seeds.
    group = load_group("long")
    return [group[f"{name}_ep{LONG_EPOCHS}_s{s}"] for s in SEEDS if f"{name}_ep{LONG_EPOCHS}_s{s}" in group]



# %% id="7KjzoXrcQLJa"
import re

# Readable names for tables and plots; the stored run names are unchanged
DISPLAY = {
    "s0_vgg": "plain VGG", "s1_bn": "+ BatchNorm", "s2_gap": "+ global average pooling",
    "s3_aug": "+ crop and flip", "s4_stage4": "+ fourth stage",
    "plain_1111": "plain, 11 layers", "res_1111": "residual, 11 layers",
    "plain_2222": "plain, 19 layers", "res_2222": "residual, 19 layers",
    "plain_3461": "plain, 31 layers", "res_3461": "residual, 31 layers",
}


def label(name):
    # Readable name of a network, a roadmap step or a recipe configuration.
    if name in DISPLAY:
        return DISPLAY[name]
    m = re.fullmatch(r"cfg(\d+)", name)
    if m:
        return f"configuration {int(m.group(1))}"
    m = re.fullmatch(r"lr([\d.e-]+)_wd([\d.e-]+)", name)
    if m:
        return f"learning rate {m.group(1)}, weight decay {m.group(2)}"
    return name


# Categorical colours in a fixed order (one per run inside a plot)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=COLORS), "lines.linewidth": 2,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def summary(runs):
    # One row per run (single seed).
    rows = [{"network": label(n), "learning rate": r["config"]["lr"], "parameters (M)": r["params"] / 1e6,
             "best validation": r["best_val_acc"], "best epoch": r["best_epoch"],
             "final training": r["final_train_acc"], "train − validation gap": r["final_train_acc"] - r["final_val_acc"],
             "test": r["test_acc"], "minutes": r["minutes"]}
            for n, r in runs.items()]
    return pd.DataFrame(rows).set_index("network").style.format(
        {"learning rate": "{:g}", "parameters (M)": "{:.2f}", "best validation": "{:.3f}", "final training": "{:.3f}",
         "train − validation gap": "{:+.3f}", "test": "{:.3f}", "minutes": "{:.1f}"})


def seed_summary(groups):
    # One row per configuration, mean ± std over the seeds. groups: {label: [runs]}
    rows = [{"network": label(name), "learning rate": runs[0]["config"]["lr"], "seeds": len(runs),
             "parameters (M)": runs[0]["params"] / 1e6,
             "validation (mean)": seed_mean(runs), "validation (std)": seed_std(runs),
             "test (mean)": seed_mean(runs, "test_acc"), "test (std)": seed_std(runs, "test_acc"),
             "final training (mean)": seed_mean(runs, "final_train_acc")}
            for name, runs in groups.items() if runs]
    return pd.DataFrame(rows).set_index("network").style.format(
        {"learning rate": "{:g}", "parameters (M)": "{:.2f}", "validation (mean)": "{:.3f}",
         "validation (std)": "{:.3f}", "test (mean)": "{:.3f}", "test (std)": "{:.3f}",
         "final training (mean)": "{:.3f}"})


def plot_curves(runs, title=""):
    # Validation accuracy, validation loss (plain CE) and training accuracy against the epoch.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for i, (name, r) in enumerate(runs.items()):
        h = r["history"]
        ep = range(1, len(h["val_acc"]) + 1)
        c = COLORS[i % len(COLORS)]
        axes[0].plot(ep, h["val_acc"], color=c, label=label(name))
        if r["best_epoch"]:
            axes[0].plot(r["best_epoch"], r["best_val_acc"], "o", color=c, markersize=7)
        axes[1].plot(ep, h["val_loss"], color=c, label=label(name))
        axes[2].plot(ep, h["train_acc"], color=c, label=label(name))
    for ax, t in zip(axes, ("validation accuracy (dot = best epoch)", "validation loss (cross-entropy)",
                            "training accuracy")):
        ax.set_title(t); ax.set_xlabel("epoch")
    axes[0].legend(fontsize=8)
    fig.suptitle(title); plt.tight_layout(); plt.show()


# %% [markdown]
# ## 1. From a plain VGG to a residual network: a roadmap
#
# Each step adds one change to the previous one and is trained at three learning rates (one seed). What we expect:
#
# - **A plain VGG** with three stages and a flatten classifier has 6.2M parameters for 3,334 training images — 5M of
#   them in the classifier, which receives 50,176 features per image, more than the number of training images. Without
#   augmentation or regularization it should **memorize** the training set: training accuracy close to 100%, validation
#   accuracy far below. Standard convolutional networks can even fit randomly shuffled labels (Zhang et al., ICLR 2017).
# - **+ BatchNorm** should make optimization better behaved and more tolerant to the learning rate.
# - **+ global average pooling** removes the 5M-weight classifier, where memorization concentrates.
# - **+ crop and flip** show a different version of every image at every epoch, so the training set can no longer be
#   memorized exactly.
# - **+ a fourth stage** adds capacity and receptive field once the network is regularized.

# %% colab={"base_uri": "https://localhost:8080/"} id="nzWzW-VHQLJY" outputId="f0273c3c-3215-4c10-a775-9fae97790954"
# [run]
for name, over in LADDER:
    run_sweep(name, "ladder", **over)
run_report()


# %% [markdown]
# ### Results
#
# Each step at its best learning rate; the curves show validation accuracy (dot = best epoch), validation loss and
# training accuracy.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Wtr32tImQLJa" outputId="300d5da5-ff4b-4a91-905f-077b83240ec6"
ladder = load_group("ladder")
steps = {n: best_of(ladder, n) for n, _ in LADDER}
steps = {n: r for n, r in steps.items() if r}
if steps:
    display(summary(steps))

    names = list(steps)
    xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(xs - 0.2, [steps[n]["best_val_acc"] for n in names], 0.38, label="validation (best epoch)", color=COLORS[0])
    ax.bar(xs + 0.2, [steps[n]["test_acc"] for n in names], 0.38, label="test", color=COLORS[1])
    for x_, n in zip(xs, names):
        ax.text(x_ - 0.2, steps[n]["best_val_acc"] + 0.01, f"{steps[n]['best_val_acc']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs, [label(n) for n in names], rotation=15, ha="right"); ax.set_ylabel("accuracy"); ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.legend(); ax.set_title("the roadmap: each step adds one change, at its best learning rate")
    plt.tight_layout(); plt.show()

    plot_curves(steps, "the roadmap")

# %% [markdown]
# **Every step improves the network**, from **18.5%** (plain VGG) to **52.6%** (fourth stage) validation accuracy: BatchNorm adds 11.7
# points, global average pooling 13.3, crop and flip 4.2 and the fourth stage 4.9.
#
# The curves show the mechanism. The first three networks reach **100% training accuracy**: they memorize the training
# set. The plain VGG does it within about 15 epochs, after which its validation loss *rises* while its training loss goes to zero —
# the network becomes confidently wrong on unseen images. BatchNorm and global average pooling make the memorizing
# network generalize better, but do not stop memorization. Augmentation does: once crop and flip are added, the training accuracy stays
# below 100% and the gap between training and validation accuracy shrinks.

# %% [markdown]
# ### Sensitivity to the learning rate
#
# Best validation accuracy of every step at each learning rate.

# %% colab={"base_uri": "https://localhost:8080/", "height": 457} id="EREoptOEQLJa" outputId="b1f921c9-419f-4102-b418-108146e1c403"
if ladder:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, _) in enumerate(LADDER):
        swept = sweep_runs(ladder, name)
        if swept:
            lrs = sorted(swept)
            ax.plot(lrs, [swept[lr]["best_val_acc"] for lr in lrs], "o-", color=COLORS[i], label=label(name), markersize=8)
    ax.set_xscale("log"); ax.set_xlabel("learning rate"); ax.set_ylabel("best validation accuracy")
    ax.set_title("every step of the roadmap across learning rates"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()

# %% [markdown]
# The plain VGG, without BatchNorm, diverges at learning rate 0.1 and reaches only 4% at 0.05; with BatchNorm every
# learning rate trains, and the best one gives 11.7 more points — the tolerance to the learning rate promised by lecture 4
# (*Batch norm: pros and cons*). After global average pooling the best learning rate moves from 0.01 to 0.05–0.1, and
# 0.01 becomes clearly worse. A single learning rate for every step would have penalized some of them: this is why
# every configuration is compared at its own best.
#
# With augmentation and a fourth stage the network is regularized and still improving: the next question is whether it
# benefits from more depth.

# %% [markdown]
# ## 2. Depth and residual connections
#
# We replicate at small scale the central experiment of the ResNet paper (lecture 5, *Residual Networks*; He et al., 2016):
# **identical plain and residual networks at increasing depth**. All six networks are the last network of the roadmap
# with a different number of blocks per stage, with and without skip connections:
#
# ![depth grid](attachment:depth_grid.png)
#
#
# - **one block per stage**: 11 convolutional layers, 4.8M parameters — the last network of the roadmap itself;
# - **two blocks per stage**: 19 layers, 11.1M — the stage layout of ResNet-18 (lecture 5, *Common variants on ImageNet*),
#   where He et al. find plain and residual networks on par;
# - **3, 4, 6 and 1 blocks**: 31 layers, 11.8M — the layout of ResNet-34 with one block in the last stage, where He et
#   al. find the plain network *worse* than its shallower version. A 512-channel block alone costs 4.7M parameters, so
#   the extra depth goes into the first three stages.
#
# Within a depth, the two networks differ only by the skip connections. lecture 5 notes that deeper plain networks show
# **higher training error**, so the problem is optimization, not overfitting: this is the signature to look for in the
# training accuracy. Each network is trained at three learning rates, then with two more seeds at its best one, since
# run-to-run variation (1–2 points) is comparable to the differences between networks.

# %% colab={"base_uri": "https://localhost:8080/"} id="eKrS-l2IQLJY" outputId="e36a9123-872c-4aaa-9d15-5833f87ec4cc"
# [run]
for name, over in GRID:
    run_sweep(name, "depth_residual", **over)
run_report()


# %% [markdown]
# Two more seeds for every network, at its best learning rate.

# %% colab={"base_uri": "https://localhost:8080/"} id="QvW1iyfAQLJY" outputId="8145b6e1-9703-4e0e-8192-054a83c68482"
# [run]
for name, over in GRID:
    run_seeds(name, "depth_residual", note="seeds", **over)
run_report()


# %% [markdown]
# ### Results
#
# The six networks with their validation accuracy (mean ± standard deviation over three seeds) and training accuracy;
# then the same numbers against depth — left validation, right training accuracy — and the training curves of the
# seed-42 runs.

# %%
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

STAGE_COLORS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]       # stages 1-4: 64, 128, 256, 512 channels
SKIP_COLOR = COLORS[1]
DEPTH_COLUMNS = [("1111", "11 layers", "1, 1, 1, 1 blocks", ""),
                 ("2222", "19 layers", "2, 2, 2, 2 blocks", "ResNet-18 layout"),
                 ("3461", "31 layers", "3, 4, 6, 1 blocks", "ResNet-34 layout, last stage cut to 1 block")]


def draw_depth_grid(values=None, title=""):
    # The six networks of the experiment: plain / residual x three depths. Each block is a small box coloured by
    # stage; residual networks have an arc (the skip connection) over every block. values: {network: [text lines]}
    grid = dict(GRID)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    col_x, cell_w, row_y = [0.0, 4.3, 8.6], 3.7, {"plain": 2.0, "res": 0.0}
    bw, bh, gap = 0.2, 0.42, 0.14
    for (key, depth, blocks, ref), x0 in zip(DEPTH_COLUMNS, col_x):
        ax.text(x0 + cell_w / 2, 4.45, depth, ha="center", fontsize=12, fontweight="bold", color="#3d3d3a")
        ax.text(x0 + cell_w / 2, 4.12, blocks + (f"\n{ref}" if ref else ""), ha="center", va="top", fontsize=9,
                color="#6b6a64")
        for kind, y0 in row_y.items():
            name = f"{kind}_{key}"
            n_blocks = grid[name]["blocks"]
            length = sum(n_blocks) * bw + (len(n_blocks) - 1) * gap
            x = x0 + (cell_w - length) / 2
            for stage, n in enumerate(n_blocks):
                for b in range(n):
                    ax.add_patch(Rectangle((x, y0 + 0.9), bw * 0.85, bh, color=STAGE_COLORS[stage]))
                    if kind == "res":
                        ax.add_patch(Arc((x + bw * 0.42, y0 + 0.9 + bh), bw * 0.95, 0.5, theta1=0, theta2=180,
                                         color=SKIP_COLOR, lw=1.4))
                    x += bw
                x += gap
            lines = values.get(name, []) if values is not None else \
                [f"{count_params(AircraftNet({**BASE, **grid[name]}, N_CLASSES)) / 1e6:.1f}M parameters"]
            for k, line in enumerate(lines):
                ax.text(x0 + cell_w / 2, y0 + 0.55 - 0.42 * k, line, ha="center", va="center",
                        fontsize=11 if k == 0 and values is not None else 9,
                        fontweight="bold" if k == 0 and values is not None else "normal",
                        color="#3d3d3a" if k == 0 else "#6b6a64")
    for kind, y0 in row_y.items():
        ax.text(-0.35, y0 + 1.1, "plain" if kind == "plain" else "residual", ha="right", va="center", fontsize=12,
                fontweight="bold", color=SKIP_COLOR if kind == "res" else "#3d3d3a")
    # the plain 11-layer network is the last network of the roadmap: framed in the colour of that step
    bottom = row_y["plain"] + (-0.08 if values is not None else 0.32)
    ax.add_patch(FancyBboxPatch((0.45, bottom), cell_w - 0.9, row_y["plain"] + 1.52 - bottom,
                                boxstyle="round,pad=0,rounding_size=0.15", facecolor="none",
                                edgecolor=COLORS[6], lw=1.4, ls="--"))
    ax.text(cell_w / 2, row_y["plain"] + 1.6, "= last network of the roadmap", ha="center", va="bottom",
            fontsize=9, color=COLORS[6])
    # how to read the grid
    ax.add_patch(FancyArrowPatch((0.3, -0.35), (12.0, -0.35), arrowstyle="-|>", mutation_scale=14, color="#9b9990"))
    ax.text(6.15, -0.55, "deeper: the effect of depth", ha="center", va="top", fontsize=9.5, color="#6b6a64")
    ax.add_patch(FancyArrowPatch((12.55, 3.3), (12.55, 1.1), arrowstyle="<|-|>", mutation_scale=14, color="#9b9990"))
    ax.text(12.75, 2.2, "same depth:\nonly the skip\nconnections\ndiffer", ha="left", va="center", fontsize=9.5,
            color="#6b6a64")
    handles = [Patch(color=c, label=f"stage {i + 1} ({ch} channels)")
               for i, (c, ch) in enumerate(zip(STAGE_COLORS, (64, 128, 256, 512)))]
    handles.append(Patch(facecolor="none", edgecolor=SKIP_COLOR, label="skip connection"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=8.5, frameon=False)
    ax.set_xlim(-1.6, 14.2); ax.set_ylim(-0.95, 4.75); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, loc="left")
    plt.tight_layout(); plt.show()



grid_runs = {n: seed_runs(n, "depth_residual") for n, _ in GRID}
grid_runs = {n: r for n, r in grid_runs.items() if r}
if len(grid_runs) == len(GRID):
    draw_depth_grid({n: [f"{100 * seed_mean(r):.1f} ± {100 * seed_std(r):.1f}",
                         f"training accuracy {100 * seed_mean(r, 'final_train_acc'):.1f}"]
                     for n, r in grid_runs.items()},
                    title="Validation accuracy (mean ± standard deviation over three seeds)")

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="RaFm2k7bQLJa" outputId="4a7e304e-7665-41ad-ad70-eea0de276bbc"
grid_runs = {n: seed_runs(n, "depth_residual") for n, _ in GRID}
grid_runs = {n: r for n, r in grid_runs.items() if r}
if grid_runs:
    display(seed_summary(grid_runs))

    convs = {n: 3 + 2 * sum(dict(GRID)[n]["blocks"]) for n in grid_runs}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for kind, kind_label, c in (("plain", "plain", COLORS[1]), ("res", "residual", COLORS[0])):
        names = [n for n in grid_runs if n.startswith(kind + "_")]
        if not names:
            continue
        for ax, key in zip(axes, ("best_val_acc", "final_train_acc")):
            ax.errorbar([convs[n] for n in names], [seed_mean(grid_runs[n], key) for n in names],
                        yerr=[np.nan_to_num(seed_std(grid_runs[n], key)) for n in names],
                        fmt="o-", color=c, capsize=4, markersize=8, label=kind_label)
    for ax, t in zip(axes, ("best validation accuracy", "final training accuracy")):
        ax.set_xlabel("convolutional layers"); ax.set_ylabel(t); ax.set_xticks(sorted(set(convs.values())))
        ax.set_title(f"{t} vs depth (mean ± std over seeds)"); ax.legend()
    plt.tight_layout(); plt.show()

    plot_curves({n: r[0] for n, r in grid_runs.items()}, "depth and residual connections (seed 42, best learning rate)")

# %% [markdown]
# **Starting point.** The plain 11-layer network is the last network of the roadmap (framed in the figure), trained
# again with the same configuration: its seed-42 runs reproduce the numbers of the roadmap exactly, since training is
# deterministic for a fixed seed. Its two extra seeds only add the standard deviation.
#
# **Degradation.** From 19 to 31 layers the plain network loses 8.3 points (55.9% → 47.6%), and its *training* accuracy
# drops too (97.5% → 91.9%): the deeper plain network is harder to optimize — the degradation problem described in lecture 5.
# The residual network improves at every depth (46.9% → 52.2% → 54.6%) and keeps fitting its training set: skip
# connections make depth usable.
#
# **Shallow networks.** At 11 and 19 layers the residual networks are about 4 points *worse* than the plain ones, well
# beyond the seed noise, while reaching a higher training accuracy (99% against 97–98%). Without any regularization,
# the ease of fitting that skip connections provide turns into more overfitting on 33 images per class. If this is the
# cause, regularization should help the residual networks more than the plain ones — the question of the next section.

# %% [markdown]
# ## 3. The training recipe
#
# We regularize the three best networks of the previous experiment and tune their training recipe, combining standard
# techniques from lecture 6 (*Regularization*):
#
# - **Weight decay** (lecture 6, *l2 regularization or weight decay*) penalizes large weights.
# - **Label smoothing** (lecture 6, *Label smoothing*; Szegedy et al., 2016) replaces the one-hot target with 1−ε on the true
#   class and ε spread over the others, so the network is not pushed towards infinite confidence.
# - **Dropout** before the classifier (lecture 6, *Dropout*), between global pooling and the last linear layer. lecture 6 notes it is
#   not used in ResNets, for its unclear interaction with BatchNorm (*Dropout: where*): zero is a real candidate.
# - **Random erasing** (lecture 6, *Cutout (Random Erasing)*; Zhong et al., 2020) hides a random rectangle of the image.
# - **Minimum crop area** of the random resized crop: smaller crops show parts of the aircraft at a larger scale
#   (lecture 6, *multi-scale training*).
# - **One-cycle schedule** (lecture 6, *Learning rate schedule: One cycle*; Smith & Topin, 2019): the learning rate rises over
#   the first 30% of training, then decays to almost zero.
#
# ![recipe search](attachment:recipe_search.png)
#
# We tune the recipe in three steps, following lecture 6 (*Random hyper-parameters search*):
#
# 1. **Grid search on learning rate x weight decay**, the two most influential parameters, log-spaced, with the other
#    parameters fixed (label smoothing 0.1, dropout 0.2, random erasing 0.25, minimum crop 50%, cosine schedule).
# 2. **Random search** on the other five parameters (Bergstra & Bengio, 2012), at the best learning rate and weight
#    decay of each network: a full grid would require 216 runs per network, so we draw 12 configurations once and
#    train the same 12 on every network.
# 3. **Confirmation with seeds**: the best of many single runs is partly the luckiest, so the two best configurations of
#    each network are retrained with two more seeds and compared on the mean.
#
# **Networks.** The 19-layer plain network and the 31-layer residual network are the two best; the 19-layer residual
# network is third but overfits the most, so it is the network regularization can help most. The 31-layer plain network
# is excluded because it does not fit its training set — an optimization problem regularization cannot fix — and the
# 11-layer networks trail the best by 5–9 points. All three
# networks receive the same tuning budget, so that the comparison remains fair.
#
# **Selection rule.** The best configuration of each network by mean validation accuracy; then the best network, or
# the simpler one if the best two are within one standard deviation.
#
# Mixup, CutMix and the exponential moving average of the weights (lecture 6) are left out: they mainly pay off with schedules
# much longer than ours.

# %% [markdown]
# ### Step 1 — learning rate and weight decay
#
# Best validation accuracy of every point of the grid (seed 42), one panel per network; the ring marks the best point.

# %% colab={"base_uri": "https://localhost:8080/"} id="T09YgdBOQLJZ" outputId="efdbbda5-052c-4e15-f47e-bebccd57ded2"
# [run]
names = finalists(TUNE_N)
if names is None:
    print("train the whole grid with all its seeds first")
else:
    print("networks:", ", ".join(label(n) for n in names))
    # 1. grid search: learning rate x weight decay, the other parameters as in RECIPE
    for name in names:
        for lr in TUNE_LRS:
            for wd in TUNE_WDS:
                run_experiment(f"{name}_lr{lr:g}_wd{wd:g}", "tune_grid", note="tuning: lr x wd",
                               **{**dict(GRID)[name], **RECIPE, "lr": lr, "weight_decay": wd})
        lr, wd = best_grid_point(name)
        print(f"==> {label(name)}: best learning rate {lr:g}, weight decay {wd:g}\n")
run_report()


# %% colab={"base_uri": "https://localhost:8080/", "height": 353} id="neNY--9cQLJa" outputId="be6ede74-60b1-438b-eb75-41c46a95663d"
FIRST_GRID_MAX_WD = 2e-3      # the first grid stopped here: the larger weight decays were added afterwards

names = finalists(TUNE_N) or []
grid_runs_t = load_group("tune_grid")
if names and grid_runs_t:
    fig, axes = plt.subplots(1, len(names), figsize=(15, 4.2), sharey=True)
    for ax, name in zip(axes, names):
        ax.axvspan(np.sqrt(FIRST_GRID_MAX_WD * 5e-3), 1.6e-2, color="#f3f2ee", zorder=0)
        ax.text(np.sqrt(5e-3 * 1e-2), 18.5, "added after\nthe first grid", ha="center", fontsize=8.5, color="#6b6a64")
        points = []
        for lr, c in zip(TUNE_LRS, ("#9b9990", COLORS[0])):
            acc = [100 * grid_runs_t.get(f"{name}_lr{lr:g}_wd{wd:g}", {}).get("best_val_acc", np.nan) for wd in TUNE_WDS]
            ax.plot(TUNE_WDS, acc, "o-", color=c, lw=2, markersize=7, label=f"learning rate {lr:g}")
            points += zip(TUNE_WDS, acc)
        wd, acc = max(points, key=lambda p: np.nan_to_num(p[1]))
        ax.plot(wd, acc, "o", markersize=16, markerfacecolor="none", markeredgecolor="#3d3d3a", markeredgewidth=1.4)
        ax.annotate(f"{acc:.1f}", (wd, acc), xytext=(0, 13), textcoords="offset points", ha="center", fontsize=9.5,
                    color="#3d3d3a")
        ax.set_xscale("log"); ax.set_xlim(7e-5, 1.6e-2); ax.set_ylim(15, 70)
        ax.set_title(label(name)); ax.set_xlabel("weight decay")
    axes[0].set_ylabel("best validation accuracy (%)"); axes[0].legend(loc="lower left", fontsize=9, frameon=False)
    plt.tight_layout(); plt.show()


# %% [markdown]
# Accuracy increases with weight decay up to **2e-3 for the plain network and 5e-3 for both residual networks**, then drops
# sharply at 1e-2. The grid initially stopped at 2e-3, which was the best value for two networks; since an optimum at
# the edge of a grid may lie beyond it, we extended the grid until accuracy decreased (shaded area), and the optimum now lies inside
# it. The residual networks, which fit their training set more easily, need the stronger regularization. With strong
# weight decay, a learning rate of 0.05 is better than 0.1.

# %% [markdown]
# ### Step 2 — random search
#
# Every configuration on every network (seed 42), sorted by the mean over the networks; in the plot every configuration is a
# line through its parameter values, ending at its mean accuracy over the three networks (blue: the two best).

# %% colab={"base_uri": "https://localhost:8080/"} id="q1-jhNX7QLJZ" outputId="9dbe68bc-468e-435c-eb65-3a9d8a78fdb7"
# [run]
# 2. random search on the other parameters, at the best lr and weight decay of each network
configs = random_configs()
for i, c in enumerate(configs):
    print(f"{label(f'cfg{i:02d}')}: {c}")
for name in finalists(TUNE_N):
    lr, wd = best_grid_point(name)
    for i, c in enumerate(configs):
        run_experiment(f"{name}_cfg{i:02d}", "tune_random", note="tuning: random search",
                       **{**dict(GRID)[name], **RECIPE, "lr": lr, "weight_decay": wd, **c})
run_report()


# %% colab={"base_uri": "https://localhost:8080/", "height": 754} id="0UkjEj6DQLJa" outputId="ce26e558-8779-42b3-9db8-beb361b5f4dd"
random_runs = load_group("tune_random")
configs = random_configs()
if names and random_runs:
    rows = []
    for i, c in enumerate(configs):
        row = {"configuration": label(f"cfg{i:02d}"), "label smoothing": c["label_smoothing"],
               "dropout": c["dropout"], "random erasing": c["random_erasing"], "minimum crop": c["crop_scale"][0],
               "schedule": c.get("schedule", "cosine")}
        for name in names:
            r = random_runs.get(f"{name}_cfg{i:02d}")
            row[label(name)] = r["best_val_acc"] if r else np.nan
        rows.append(row)
    table = pd.DataFrame(rows).set_index("configuration")
    nets = [label(n) for n in names]
    table["mean"] = table[nets].mean(axis=1)
    display(table.sort_values("mean", ascending=False).style.format({n: "{:.3f}" for n in nets + ["mean"]}))

    # parallel coordinates: one vertical axis per parameter, the last one for the mean accuracy
    params = ["label smoothing", "dropout", "random erasing", "minimum crop", "schedule"]
    levels = {p: sorted(table[p].unique(), key=str) for p in params}
    lo, hi = np.floor(table["mean"].min() * 100) / 100, np.ceil(table["mean"].max() * 100) / 100
    order = table.sort_values("mean")                       # the two most accurate, in blue, are drawn last
    tag = dict(fc="white", ec="none", pad=1.5)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for rank, (cfg, row) in enumerate(order.iterrows()):
        shift = (rank - (len(order) - 1) / 2) * 0.012       # separates the lines that share a value
        ys = [levels[p].index(row[p]) / (len(levels[p]) - 1) + shift for p in params]
        ys.append((row["mean"] - lo) / (hi - lo))
        top = rank >= len(order) - 2
        ax.plot(range(len(ys)), ys, color=(COLORS[0] if rank == len(order) - 1 else "#86b6ef") if top else "#c9c7bd",
                lw=2.6 if top else 1.3, zorder=3 + rank)
        if top or rank == 0:
            ax.text(len(params) + 0.06, ys[-1], f"{cfg} ({100 * row['mean']:.1f})", va="center", fontsize=8.5,
                    color="#3d3d3a")
    for k, p in enumerate(params):
        ax.axvline(k, color="#c9c7bd", lw=1, zorder=1)
        for j, v in enumerate(levels[p]):
            text = ("one-cycle" if v == "onecycle" else f"{v:.0%}" if p == "minimum crop" else
                    f"{v:g}" if isinstance(v, float) else str(v))
            ax.text(k - 0.04, j / (len(levels[p]) - 1), text, ha="right", va="center", fontsize=8.5, color="#6b6a64",
                    bbox=tag, zorder=20)
    ax.axvline(len(params), color="#c9c7bd", lw=1, zorder=1)
    for v in np.arange(np.ceil(lo * 20) / 20, hi, 0.05):
        ax.text(len(params) - 0.04, (v - lo) / (hi - lo), f"{100 * v:.0f}", ha="right", va="center", fontsize=8.5,
                color="#6b6a64", bbox=tag, zorder=20)
    ax.set_xticks(range(len(params) + 1), params + ["mean validation\naccuracy (%)"])
    ax.set_yticks([]); ax.set_xlim(-0.5, len(params) + 1.1); ax.set_ylim(-0.08, 1.08)
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False); ax.grid(False)
    ax.tick_params(axis="x", length=0)
    plt.tight_layout(); plt.show()


# %% [markdown]
# The ranking is consistent across networks: the same configuration (configuration 5, the dark blue line) — **label smoothing 0.2, no dropout,
# random erasing 0.25, minimum crop 50%, one-cycle schedule** — is the best or second best for all three. The one-cycle schedule
# appears in the best configurations, no dropout in the two best — consistent with lecture 6's remark on dropout and
# BatchNorm — and random erasing at 0.5 in all of the worst ones. Since each run changes several parameters at once,
# these are tendencies; the ablation study measures each component in isolation.

# %% [markdown]
# ### Step 3 — confirmation with seeds
#
# The two best configurations of each network, mean ± standard deviation over three seeds.

# %% colab={"base_uri": "https://localhost:8080/"} id="0rA8vdCCQLJZ" outputId="f238778b-15ba-41ab-b1f1-b40f42a1ed97"
# [run]
# 3. confirmation: the TOP_K best configurations of each network, with the other seeds
for name in finalists(TUNE_N):
    for run in top_configs(name):
        for seed in SEEDS[1:]:
            run_experiment(f"{run['name']}_s{seed}", "tune_seeds", note="tuning: seeds",
                           **{**run_overrides(run), "seed": seed})
run_report()


# %% colab={"base_uri": "https://localhost:8080/", "height": 755} id="wwg-HozpQLJb" outputId="d9904fa7-5afe-40a9-d874-f876706e0cd2"
if names:
    groups, best_per_net = {}, {}
    for name in names:
        conf = {n: runs for n, runs in confirmed(name).items() if len(runs) == len(SEEDS)}
        for n, runs in conf.items():
            groups[f"{label(name)} | {label(n.removeprefix(name + '_'))}"] = runs
        if conf:
            best = max(conf, key=lambda n: seed_mean(conf[n]))
            best_per_net[name] = (best, conf[best])
    if groups:
        display(seed_summary(groups))

    if len(best_per_net) == len(names):
        ranked = sorted(best_per_net, key=lambda n: -seed_mean(best_per_net[n][1]))
        for n in ranked:
            cfg_name, runs = best_per_net[n]
            print(f"{label(n):20s} best: {label(cfg_name.removeprefix(n + '_'))} | validation {seed_mean(runs):.3f} ± {seed_std(runs):.3f} | "
                  f"test {seed_mean(runs, 'test_acc'):.3f} ± {seed_std(runs, 'test_acc'):.3f}")
        first, second = ranked[0], ranked[1]
        diff = seed_mean(best_per_net[first][1]) - seed_mean(best_per_net[second][1])
        noise = max(seed_std(best_per_net[first][1]), seed_std(best_per_net[second][1]))
        if diff > noise:
            final = first
            print(f"-> {label(first)}: better by {diff:.3f}, more than one standard deviation ({noise:.3f})")
        else:
            final = min((first, second), key=simplicity)
            print(f"-> a tie ({diff:.3f} <= {noise:.3f}): the simpler network, {label(final)}")
        print("final configuration:", run_overrides(best_per_net[final][1][0]))
        plot_curves({f"{label(n)} ({label(best_per_net[n][0].removeprefix(n + '_'))})": best_per_net[n][1][0] for n in ranked},
                    "best tuned configuration of each network (seed 42)")

# %% [markdown]
# All three networks select that same configuration, which brings them to 63–64% validation accuracy, 7–12 points above their
# unregularized versions. Tuning confirms the hypothesis of the depth experiment: without regularization
# the 19-layer residual network was 3.7 points *worse* than the plain one; with the tuned recipe it is 1.1 points
# *better*. The ease of fitting given by the skip connections turns into accuracy once overfitting is controlled.
#
# At 70 epochs the two residual networks are within 0.1 points, so our rule would select the simpler one. However, all
# three networks reach their best validation accuracy in the last epochs, with a training accuracy still around 90%:
# they are still improving when training stops.
#
# **Limitation.** The search is staged — architecture first, then recipe; learning rate and weight decay before the
# other parameters — so it assumes the effects to be roughly independent. The flip in the ranking of the 19-layer plain and
# residual networks shows that they are not entirely: this is why we tuned three networks rather than one, and why the ablation
# measures every component on the final model.

# %% [markdown]
# ## 4. Longer training and the final model
#
# We train the three tuned networks again for **150 epochs**, each with its best configuration and no other change,
# with three seeds each. The one-cycle schedule is defined in fractions of the training, so it stretches with the number
# of epochs: in the middle of a longer run the learning rate is still high, and validation accuracy only rises during
# the final decay. Since the recipe was tuned for 70 epochs, a longer training also gives more time to overfit: we check
# that the gap between training and validation accuracy does not grow. The final model is selected with the same rule.

# %% colab={"base_uri": "https://localhost:8080/"} id="eJj0cSoOQLJZ" outputId="28f103dc-8786-4326-e9ec-7d6b69800de6"
# [run]
for name in finalists(TUNE_N):
    best = best_tuned(name)
    if best is None:
        print(f"{label(name)}: train the recipe search with all its seeds first")
        continue
    cfg_name, runs = best
    print(f"{label(name)}: {label(cfg_name.removeprefix(name + '_'))}, {LONG_EPOCHS} epochs")
    for seed in SEEDS:
        run_experiment(f"{name}_ep{LONG_EPOCHS}_s{seed}", "long", note=f"{cfg_name}, {LONG_EPOCHS} epochs",
                       **{**run_overrides(runs[0]), "epochs": LONG_EPOCHS, "seed": seed})
run_report()


# %% [markdown]
# ### Results
#
# Each tuned network at 70 and at 150 epochs (mean ± standard deviation over three seeds) and the selection rule at 150
# epochs; the curves compare the two budgets (seed 42).

# %% colab={"base_uri": "https://localhost:8080/", "height": 755} id="fG8OmpbpQLJb" outputId="fa7b56cd-10bf-4316-8d20-9c0dae878b85"
names = finalists(TUNE_N) or []
groups, long_final = {}, {}
for name in names:
    best, runs = best_tuned(name) or (None, [])
    if runs:
        groups[f"{label(name)} | {BASE['epochs']} epochs"] = runs
    long = long_runs(name)
    if long:
        groups[f"{label(name)} | {LONG_EPOCHS} epochs"] = long
    if len(long) == len(SEEDS):
        long_final[name] = long
if groups:
    display(seed_summary(groups))

if names and len(long_final) == len(names):
    ranked = sorted(long_final, key=lambda n: -seed_mean(long_final[n]))
    for n in ranked:
        runs = long_final[n]
        print(f"{label(n):20s} {LONG_EPOCHS} epochs: validation {seed_mean(runs):.3f} ± {seed_std(runs):.3f} | "
              f"test {seed_mean(runs, 'test_acc'):.3f} ± {seed_std(runs, 'test_acc'):.3f} | "
              f"train {seed_mean(runs, 'final_train_acc'):.3f}")
    first, second = ranked[0], ranked[1]
    diff = seed_mean(long_final[first]) - seed_mean(long_final[second])
    noise = max(seed_std(long_final[first]), seed_std(long_final[second]))
    if diff > noise:
        final = first
        print(f"-> {label(first)}: better by {diff:.3f}, more than one standard deviation ({noise:.3f})")
    else:
        final = min((first, second), key=simplicity)
        print(f"-> a tie ({diff:.3f} <= {noise:.3f}): the simpler network, {label(final)}")
    print("final configuration:", run_overrides(long_final[final][0]))
    curves = {}
    for n in ranked:
        curves[f"{label(n)} | {BASE['epochs']} epochs"] = best_tuned(n)[1][0]
        curves[f"{label(n)} | {LONG_EPOCHS} epochs"] = long_final[n][0]
    plot_curves(curves, f"tuned networks: {BASE['epochs']} vs {LONG_EPOCHS} epochs (seed 42)")

# %% [markdown]
# **Longer training adds 7–10 points** to every network, with no other change: at 70 epochs they were stopped too early.
# **Overfitting does not increase**: the train–validation gap of the 31-layer residual network goes from 26 points at 70 epochs to 24 at
# 150, so the recipe tuned for 70 epochs still controls overfitting with a longer budget.
#
# **The ranking changes.** At 150 epochs the 31-layer residual network is 1.6 points better than the second network, more than twice the
# run-to-run noise, so by our selection rule it is the final model: the deepest residual network benefits most from the
# additional training. Training only the winner at 70 epochs longer would have missed it.
#
# **Final model: the 31-layer residual network** (stem of three 3x3 convolutions; four
# stages of 3, 4, 6 and 1 basic blocks with 64, 128, 256 and 512 channels; global average pooling; linear classifier;
# 11.9M parameters), trained for 150 epochs with SGD (momentum 0.9), a one-cycle schedule peaking at a learning rate of 0.05, weight
# decay 5e-3, label smoothing 0.2, random resized crops of at least 50% of the image, horizontal flips and random
# erasing (p = 0.25): **74.0% ± 0.7 validation and 74.2% ± 0.2 test accuracy**. It combines every component justified
# by the previous experiments — BatchNorm, global average pooling, augmentation, a fourth stage, depth and residual
# connections — with a tuned training recipe.

# %% [markdown]
# ## 5. Error analysis of the final model
#
# Where does the final model fail? We use the test predictions stored with each run: the per-class accuracy of the
# three seeds and the pairs of variants the model confuses most often (seed 42). Class names are shown when the dataset
# is available.

# %%
names = finalists(TUNE_N) or []
long_final = {n: long_runs(n) for n in names if len(long_runs(n)) == len(SEEDS)}
if not names or len(long_final) < len(names):
    print("train the longer runs first")
else:
    final = max(long_final, key=lambda n: seed_mean(long_final[n]))
    ranked = sorted(long_final, key=lambda n: -seed_mean(long_final[n]))
    if seed_mean(long_final[ranked[0]]) - seed_mean(long_final[ranked[1]]) <= max(seed_std(long_final[ranked[0]]), seed_std(long_final[ranked[1]])):
        final = min(ranked[:2], key=simplicity)
    names_of = globals().get("CLASSES") or [f"class {i}" for i in range(N_CLASSES)]
    runs = long_final[final]

    # per-class test accuracy, mean over the seeds
    per_class = []
    for r in runs:
        true, pred = np.array(r["test_true"]), np.array(r["test_pred"])
        per_class.append(np.bincount(true, weights=(pred == true), minlength=N_CLASSES) / np.bincount(true, minlength=N_CLASSES))
    per_class = np.mean(per_class, axis=0)

    # most frequent confusions (seed 42)
    true, pred = np.array(runs[0]["test_true"]), np.array(runs[0]["test_pred"])
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(cm, (true, pred), 1)
    off = cm.copy(); np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1][:10], off.shape))[0]
    display(pd.DataFrame([dict(true=names_of[t], predicted=names_of[p], errors=off[t, p],
                               share_of_class=off[t, p] / cm[t].sum()) for t, p in top])
            .style.format({"share_of_class": "{:.0%}"}))

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(per_class, bins=np.linspace(0, 1, 21), color=COLORS[0], edgecolor="white")
    ax.axvline(per_class.mean(), color=COLORS[1], ls="--", lw=1.5, label=f"mean {per_class.mean():.1%}")
    ax.set_xlabel("test accuracy of the class (mean over seeds)"); ax.set_ylabel("number of classes")
    ax.set_title(f"{label(final)}: per-class test accuracy"); ax.legend()
    plt.tight_layout(); plt.show()
    worst = np.argsort(per_class)[:5]
    print("hardest classes:", ", ".join(f"{names_of[i]} ({per_class[i]:.0%})" for i in worst))

# %% [markdown]
# A fine-grained task is expected to concentrate its errors on visually similar variants rather than spreading them
# uniformly: the pairs above show which variants the model cannot yet separate, and the per-class accuracy shows how
# unevenly the 100 classes are learned.

# %% [markdown]
# ## 6. Reproducibility
#
# Training is deterministic for a given seed, hardware and software. The study contains duplicate trainings of
# identical configurations: the last network of the roadmap and the 11-layer plain network of the depth experiment are the same
# network trained separately, at three learning rates. We compare their stored results; optionally (`CHECK_RETRAIN = True`),
# the final model is also trained again and compared with its stored run.

# %%
CHECK_RETRAIN = False      # True: also retrain the final model (seed 42) and compare it with its stored run

def same_run(a, b):
    cfg = lambda r: {k: v for k, v in r["config"].items() if k not in ("name", "group")}
    return dict(same_config=cfg(a) == cfg(b),
                identical_curves=all(a["history"][k] == b["history"][k] for k in ("train_loss", "train_acc", "val_loss", "val_acc")),
                identical_test_predictions=a["test_pred"] == b["test_pred"],
                val=a["best_val_acc"], test=a["test_acc"], device=a["device"])

rows = []
for lr in LR_SWEEP:
    a = json.loads(result_path(f"s4_stage4_lr{lr:g}", "ladder", SEEDS[0]).read_text())
    b = json.loads(result_path(f"plain_1111_lr{lr:g}", "depth_residual", SEEDS[0]).read_text())
    rows.append(dict(pair=f"last roadmap network vs 11-layer plain network, learning rate {lr:g}", **same_run(a, b)))

if CHECK_RETRAIN:
    if data is None:
        print("skipped: the dataset is not available")
    else:
        names = finalists(TUNE_N) or []
        long_final = {n: long_runs(n) for n in names if len(long_runs(n)) == len(SEEDS)}
        final = max(long_final, key=lambda n: seed_mean(long_final[n]))
        stored = long_final[final][0]
        cfg = {k: v for k, v in stored["config"].items() if k not in ("name", "group")}
        fix_random(cfg["seed"])                          # exactly as run_experiment does
        model, hist, best_epoch, _ = fit(cfg, data, verbose=False)
        _, test_acc, test_pred = evaluate(model, data["test"], data, cfg, return_preds=True)
        fresh = dict(config=stored["config"], history=hist, best_val_acc=max(hist["val_acc"]), test_acc=test_acc,
                     test_pred=test_pred, device=torch.cuda.get_device_name() if AMP else str(DEVICE))
        rows.append(dict(pair=f"stored {label(final)} (seed 42) vs retrained now", **same_run(stored, fresh)))

display(pd.DataFrame(rows).set_index("pair").style.format({"val": "{:.4f}", "test": "{:.4f}"}))

# %% [markdown]
# The duplicate trainings are identical in every epoch and in every test prediction: the stored results can be
# reproduced exactly on the same hardware (all runs were trained on an NVIDIA A100) and software. On different GPUs
# or library versions, floating-point operations are executed in a different order and results differ slightly, by
# less than the run-to-run noise measured with the seeds.

# %% [markdown]
# ## 7. Ablation study
#
# *To be added: leave-one-out ablation of the final model.*

# %% [markdown]
# ## Conclusions
#
# - A plain CNN memorizes the 33 images per class of FGVC-Aircraft; BatchNorm, global average pooling and augmentation
#   turn memorization into generalization (18.5% → 47.7%), and a fourth stage adds capacity (52.6%).
# - Deeper plain networks degrade, while residual networks keep improving with depth — but without regularization the
#   ease of fitting of residual networks becomes overfitting.
# - A tuned training recipe (strong weight decay, label smoothing, random erasing, one-cycle schedule) reverses that
#   ranking and brings every network to 63–64%; training for 150 epochs adds 7–10 more points.
# - The final model, a 31-layer residual network, reaches **74.2% ± 0.2 test accuracy**, with every component justified
#   by a measurement.

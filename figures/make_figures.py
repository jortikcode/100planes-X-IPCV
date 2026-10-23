"""Generate the diagrams embedded in code/part1.ipynb from the stored results (results/part1).

    python figures/make_figures.py

Writes figures/roadmap.png (introduction), figures/data_pipeline.png (Data) figures/depth_grid.png (section 2),
figures/network.png (the network) and figures/recipe_search.png (section 3).
Only the stored JSON results and, for the data figure, one image of the dataset are read: no training, no PyTorch.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "part1"
OUT = ROOT / "figures"

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#9b9990", "#3d3d3a", "#6b6a64"
STAGE_COLORS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]       # stages 1-4: 64, 128, 256, 512 channels
# One colour per component added by the roadmap, shared by the roadmap and the network diagram
STEP = {"base": "#9b9990", "bn": "#2a78d6", "gap": "#1baf7a", "aug": "#eda100", "stage4": "#6250d6",
        "depth": ORANGE, "training": "#5f5e5a"}
LRS, SEEDS = (0.1, 0.05, 0.01), (42, 43, 44)
GRID = {"plain_1111": (1, 1, 1, 1), "res_1111": (1, 1, 1, 1), "plain_2222": (2, 2, 2, 2),
        "res_2222": (2, 2, 2, 2), "plain_3461": (3, 4, 6, 1), "res_3461": (3, 4, 6, 1)}
FINAL = "res_3461"


def load(group, name, seed=42):
    return json.loads((RESULTS / group / f"{name}_seed{seed}.json").read_text())


def best_lr_run(group, name):
    # The seed-42 run at the best of the three learning rates.
    return max((load(group, f"{name}_lr{lr:g}") for lr in LRS), key=lambda r: r["best_val_acc"])


def three_seeds(name):
    # A network of the depth experiment: seed 42 at its best learning rate, plus seeds 43 and 44 at that rate.
    best = best_lr_run("depth_residual", name)
    lr = best["config"]["lr"]
    return [best] + [load("depth_residual_seeds", f"{name}_lr{lr:g}_s{s}", s) for s in SEEDS[1:]]


def mean_std(runs, key="best_val_acc"):
    v = [100 * r[key] for r in runs]
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None)


def roadmap():
    tuned = [load("tune_random", f"{FINAL}_cfg05")] + [load("tune_seeds", f"{FINAL}_cfg05_s{s}", s) for s in SEEDS[1:]]
    long = [load("long", f"{FINAL}_ep150_s{s}", s) for s in SEEDS]
    rows = [
        ("Architecture", None, None),
        ("plain VGG", [best_lr_run("ladder", "s0_vgg")], [], STEP["base"]),
        ("+ BatchNorm", [best_lr_run("ladder", "s1_bn")], [], STEP["bn"]),
        ("+ global average pooling", [best_lr_run("ladder", "s2_gap")], [], STEP["gap"]),
        ("+ crop and flip", [best_lr_run("ladder", "s3_aug")], [], STEP["aug"]),
        ("+ fourth stage", [best_lr_run("ladder", "s4_stage4")], [], STEP["stage4"]),
        ("Depth and residual connections", None, None),
        ("+ depth and skip connections\n(selected among 6 networks)", three_seeds(FINAL),
         [three_seeds(n) for n in GRID if n != FINAL], STEP["depth"]),
        ("Training recipe", None, None),
        ("+ tuned recipe\n(best of the search)", tuned, [], STEP["training"]),
        ("Longer training", None, None),
        ("+ 150 epochs", long, [], STEP["training"]),
    ]
    fig, ax = plt.subplots(figsize=(10, 0.46 * len(rows) + 1.2))
    for y, (text, runs, others, *color) in enumerate(rows):
        if runs is None:
            ax.text(-1.5, y, text, ha="right", va="center", fontsize=10, fontweight="bold", color=INK)
            continue
        acc, err = mean_std(runs)
        ax.barh(y, acc, height=0.62, color=color[0], edgecolor="white", linewidth=0.8, zorder=2)
        if err is not None:
            ax.errorbar(acc, y, xerr=err, color=INK, capsize=3, lw=1, zorder=3)
        alt = [mean_std(o)[0] for o in others]
        if alt:     # the other networks of the experiment, on a thin line just below the bar
            ax.plot([min(alt), max(alt)], [y + 0.48] * 2, color="#c9c7bd", lw=1, zorder=1)
            ax.plot(alt, [y + 0.48] * len(alt), "o", color=GREY, markersize=5, markeredgecolor="white", zorder=4)
        ax.text(acc + (err or 0) + 0.8, y, f"{acc:.1f}" + (f" ± {err:.1f}" if err is not None else ""),
                va="center", fontsize=9, color=INK)
        ax.text(-1.5, y, text, ha="right", va="center", fontsize=9.5, color=INK)
        ax.text(1.03, y, f"{runs[0]['params'] / 1e6:.1f}M", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=9, color=MUTED)
    ax.axvline(50, color=GREY, ls="--", lw=1, zorder=1)
    ax.text(50, -0.9, "target 50%", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.text(1.03, -0.9, "parameters", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=8.5,
            color=MUTED)
    ax.set_xlim(0, 85); ax.set_ylim(len(rows) - 0.4, -1.2)
    ax.set_yticks([]); ax.set_xticks(range(0, 81, 10)); ax.set_xlabel("validation accuracy (%)")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["left", "top", "right"]].set_visible(False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=GREY, markeredgecolor="white", markersize=7,
                               label="other networks of the experiment")],
               loc="upper right", bbox_to_anchor=(0.97, 0.99), fontsize=8.5, frameon=False)
    fig.suptitle("From a plain VGG to the final model", fontsize=12, x=0.02, ha="left")
    fig.subplots_adjust(left=0.30, right=0.89, top=0.92, bottom=0.09)
    fig.savefig(OUT / "roadmap.png", dpi=150, bbox_inches="tight")


def depth_grid():
    params = {n: best_lr_run("depth_residual", n)["params"] for n in GRID}
    columns = [("1111", "11 layers", "1, 1, 1, 1 blocks", ""),
               ("2222", "19 layers", "2, 2, 2, 2 blocks", "ResNet-18 layout"),
               ("3461", "31 layers", "3, 4, 6, 1 blocks", "ResNet-34 layout, last stage cut to 1 block")]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    col_x, cell_w, row_y = [0.0, 4.3, 8.6], 3.7, {"plain": 2.0, "res": 0.0}
    bw, bh, gap = 0.2, 0.42, 0.14
    for (key, depth, blocks, ref), x0 in zip(columns, col_x):
        ax.text(x0 + cell_w / 2, 4.45, depth, ha="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x0 + cell_w / 2, 4.12, blocks + (f"\n{ref}" if ref else ""), ha="center", va="top", fontsize=9,
                color=MUTED)
        for kind, y0 in row_y.items():
            name = f"{kind}_{key}"
            n_blocks = GRID[name]
            x = x0 + (cell_w - (sum(n_blocks) * bw + (len(n_blocks) - 1) * gap)) / 2
            for stage, n in enumerate(n_blocks):
                for _ in range(n):
                    ax.add_patch(Rectangle((x, y0 + 0.9), bw * 0.85, bh, color=STAGE_COLORS[stage]))
                    if kind == "res":       # the skip connection around the block
                        ax.add_patch(Arc((x + bw * 0.42, y0 + 0.9 + bh), bw * 0.95, 0.5, theta1=0, theta2=180,
                                         color=ORANGE, lw=1.4))
                    x += bw
                x += gap
            ax.text(x0 + cell_w / 2, y0 + 0.55, f"{params[name] / 1e6:.1f}M parameters", ha="center", va="center",
                    fontsize=9, color=INK)
    # the plain 11-layer network is the last network of the roadmap: framed in the colour of that step
    ax.add_patch(FancyBboxPatch((0.75, row_y["plain"] + 0.32), cell_w - 1.5, 1.2, boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor="none", edgecolor=STEP["stage4"], lw=1.4, ls="--"))
    ax.text(cell_w / 2, row_y["plain"] + 1.6, "= last network of the roadmap", ha="center", va="bottom", fontsize=9,
            color=STEP["stage4"])
    for kind, y0 in row_y.items():
        ax.text(-0.35, y0 + 1.1, "plain" if kind == "plain" else "residual", ha="right", va="center", fontsize=12,
                fontweight="bold", color=ORANGE if kind == "res" else INK)
    ax.add_patch(FancyArrowPatch((0.3, -0.35), (12.0, -0.35), arrowstyle="-|>", mutation_scale=14, color=GREY))
    ax.text(6.15, -0.55, "deeper: the effect of depth", ha="center", va="top", fontsize=9.5, color=MUTED)
    ax.add_patch(FancyArrowPatch((12.55, 3.3), (12.55, 1.1), arrowstyle="<|-|>", mutation_scale=14, color=GREY))
    ax.text(12.75, 2.2, "same depth:\nonly the skip\nconnections\ndiffer", ha="left", va="center", fontsize=9.5,
            color=MUTED)
    handles = [Patch(color=c, label=f"stage {i + 1} ({ch} channels)")
               for i, (c, ch) in enumerate(zip(STAGE_COLORS, (64, 128, 256, 512)))]
    handles.append(Patch(facecolor="none", edgecolor=ORANGE, label="skip connection"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=8.5, frameon=False)
    ax.set_xlim(-1.6, 14.2); ax.set_ylim(-0.95, 4.75); ax.axis("off")
    ax.set_title("Six networks, all starting from the last network of the roadmap", fontsize=12, loc="left")
    plt.tight_layout()
    fig.savefig(OUT / "depth_grid.png", dpi=150, bbox_inches="tight")


def data_pipeline(image_id="0237111", dataset=Path.home() / "data" / "fgvc" / "fgvc-aircraft-2013b"):
    # One training image through the preprocessing (once per split) and the augmentation (every batch). Needs the
    # FGVC-Aircraft archive (the folder that torchvision.datasets.FGVCAircraft downloads).
    from PIL import Image
    size, banner = 224, 20
    original = Image.open(dataset / "data" / "images" / f"{image_id}.jpg").convert("RGB")
    W, H = original.size
    cropped = original.crop((0, 0, W, H - banner))
    h = H - banner
    side = h * size / (size * 8 // 7)                 # usual recipe: short side to 256, centre crop 224
    left = (W - side) / 2
    centre = cropped.crop((round(left), round((h - side) / 2), round(left + side), round((h + side) / 2))).resize((size, size))
    stretch = cropped.resize((size, size), Image.BILINEAR)

    fig = plt.figure(figsize=(14, 7.4))
    top = fig.add_gridspec(1, 4, left=0.02, right=0.98, top=0.86, bottom=0.53, wspace=0.08,
                           width_ratios=[W / H, W / h, 1, 1])
    bottom = fig.add_gridspec(1, 5, left=0.02, right=0.98, top=0.37, bottom=0.03, wspace=0.08)
    fig.text(0.02, 0.97, "Preprocessing — once per split", fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.47, "Augmentation — every training batch, different at every epoch", fontsize=12,
             fontweight="bold", color=INK, va="top")

    def panel(spec, img, title, edge=None):
        ax = fig.add_subplot(spec)
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(edge is not None); s.set_color(edge or "none"); s.set_linewidth(2.5)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        return ax

    ax = panel(top[0], original, f"original ({W} × {H})")
    ax.add_patch(Rectangle((0, h), W, banner, facecolor=ORANGE, alpha=0.45, edgecolor=ORANGE, lw=1.5))
    ax.text(W / 2, h - 8, "copyright banner: removed", ha="center", va="bottom", fontsize=9, color="white",
            fontweight="bold", bbox=dict(facecolor=ORANGE, edgecolor="none", pad=2))
    ax = panel(top[1], cropped, "usual recipe: resize + centre crop")
    for x0, x1 in ((0, left), (left + side, W)):
        ax.add_patch(Rectangle((x0, 0), x1 - x0, h, facecolor="black", alpha=0.55, edgecolor="none"))
    ax.add_patch(Rectangle((left, (h - side) / 2), side, side, fill=False, edgecolor="white", ls="--", lw=1.5))
    for x, word in ((left / 2, "nose\nlost"), (left + side + (W - left - side) / 2, "tail\nlost")):
        ax.text(x, h / 2, word, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    panel(top[2], centre, "centre crop: nose and tail cut", edge=GREY)
    panel(top[3], stretch, "ours: stretch, whole aircraft", edge=BLUE)

    rng = np.random.default_rng(3)
    mean = tuple(int(255 * m) for m in (0.488, 0.519, 0.544))      # random erasing paints the mean colour
    for spec, (flip, erase) in zip(bottom, [(False, False), (True, False), (False, True), (True, False), (True, True)]):
        while True:     # RandomResizedCrop: 50-100% of the area, aspect ratio 3:4-4:3
            area = rng.uniform(0.5, 1.0) * size * size
            r = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
            w, hh = int(round(np.sqrt(area * r))), int(round(np.sqrt(area / r)))
            if w <= size and hh <= size:
                break
        x, y = rng.integers(0, size - w + 1), rng.integers(0, size - hh + 1)
        img = stretch.crop((x, y, x + w, y + hh)).resize((size, size), Image.BILINEAR)
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if erase:       # a random rectangle of 2-33% of the image
            a = rng.uniform(0.02, 0.33) * size * size
            q = np.exp(rng.uniform(np.log(0.3), np.log(3.3)))
            ew, eh = min(int(np.sqrt(a / q)), size), min(int(np.sqrt(a * q)), size)
            ex, ey = rng.integers(0, size - ew + 1), rng.integers(0, size - eh + 1)
            arr = np.array(img); arr[ey:ey + eh, ex:ex + ew] = mean; img = Image.fromarray(arr)
        panel(spec, img, " + ".join(["random crop"] + (["flip"] if flip else []) + (["erasing"] if erase else [])))
    fig.savefig(OUT / "data_pipeline.png", dpi=150, bbox_inches="tight")


def network():
    # The network built by the study, with each component coloured as the roadmap step that adds it.
    fig, ax = plt.subplots(figsize=(13, 7.2))

    def box(x, y, w, h, text, color=None, fontsize=10):
        face = (color + "26") if color else "#f3f2ee"           # light fill of the step colour
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=face, edgecolor=color or "#c9c7bd", lw=1.8 if color else 1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK)

    def arrow(x0, y0, x1, y1, color="#9b9990"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3))

    # --- the network, top to bottom
    x, w, h, step = 0.0, 5.0, 0.62, 0.95
    layers = [("image  3 × 224 × 224", None, ""),
              ("stem: three 3×3 convolutions + max pooling", None, "56 × 56"),
              ("stage 1  ·  64 channels", None, "56 × 56"),
              ("stage 2  ·  128 channels", None, "28 × 28"),
              ("stage 3  ·  256 channels", None, "14 × 14"),
              ("stage 4  ·  512 channels", STEP["stage4"], "7 × 7"),
              ("global average pooling", STEP["gap"], "512"),
              ("linear  →  100 classes", None, "")]
    ys = [7.0 - i * step for i in range(len(layers))]
    for (text, color, size), y in zip(layers, ys):
        box(x, y, w, h, text, color)
        if size:
            ax.text(x + w + 0.15, y + h / 2, size, va="center", fontsize=9, color=MUTED)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(x + w / 2, y0, x + w / 2, y1 + h)
    for k, y in enumerate(ys[2:6]):      # every stage repeats the block n_k times
        ax.text(x - 0.15, y + h / 2, f"$n_{k + 1}$ ×", ha="right", va="center", fontsize=10.5, color=STEP["depth"],
                fontweight="bold")
    ax.text(x + w + 0.95, ys[6] + h / 2, "(plain VGG: flatten)", va="center", fontsize=8.5, color=MUTED)

    # --- the block, left to right
    bx, by = 7.7, 3.75
    ax.text(bx, by + 1.9, "every block of a stage", fontsize=11, fontweight="bold", color=INK)
    parts = [("3×3\nconv", None), ("Batch\nNorm", STEP["bn"]), ("ReLU", None), ("3×3\nconv", None),
             ("Batch\nNorm", STEP["bn"])]
    bw, gap_ = 0.82, 0.32
    ax.text(bx, by + 0.31, "$x$", ha="center", va="center", fontsize=12, color=INK)
    xs = [bx + 0.45 + i * (bw + gap_) for i in range(len(parts))]
    for (text, color), xx in zip(parts, xs):
        box(xx, by, bw, h, text, color, fontsize=9)
    arrow(bx + 0.15, by + h / 2, xs[0], by + h / 2)
    for xa, xb in zip(xs[:-1], xs[1:]):
        arrow(xa + bw, by + h / 2, xb, by + h / 2)
    px = xs[-1] + bw + 0.55
    ax.add_patch(plt.Circle((px, by + h / 2), 0.22, facecolor="white", edgecolor=STEP["depth"], lw=1.8))
    ax.text(px, by + h / 2, "+", ha="center", va="center", fontsize=13, color=STEP["depth"])
    arrow(xs[-1] + bw, by + h / 2, px - 0.22, by + h / 2)
    box(px + 0.5, by, bw, h, "ReLU", None, fontsize=9)
    arrow(px + 0.22, by + h / 2, px + 0.5, by + h / 2)
    # the skip connection, above the block
    sy = by + 1.25
    ax.plot([bx, bx, px, px], [by + h / 2 + 0.2, sy, sy, by + h / 2 + 0.22], color=STEP["depth"], lw=2)
    ax.text((bx + px) / 2, sy + 0.12, "skip connection: identity, or 1×1 convolution when the shape changes",
            ha="center", va="bottom", fontsize=9, color=STEP["depth"])

    # --- the link between a stage and its blocks, and the final model
    ax.add_patch(FancyArrowPatch((x + w + 1.05, ys[3] + h / 2), (bx - 0.25, by + h / 2), arrowstyle="-|>",
                                 mutation_scale=12, color="#c9c7bd", lw=1.3, ls="--"))
    ax.text(bx, by - 0.75, "final model: $n$ = 3, 4, 6, 1 blocks  →  3 + 2 · 14 = 31 convolutional layers, "
            "11.9M parameters", fontsize=9.5, color=INK)

    handles = [Patch(facecolor=STEP[k] + "26", edgecolor=STEP[k], lw=1.8, label=t) for k, t in
               (("bn", "+ BatchNorm"), ("gap", "+ global average pooling"), ("stage4", "+ fourth stage"),
                ("depth", "+ depth and skip connections"))]
    ax.legend(handles=handles, title="added by the roadmap step", title_fontsize=9, loc="lower right",
              bbox_to_anchor=(1.0, 0.0), fontsize=9, frameon=False)
    ax.set_xlim(-1.1, 16.6); ax.set_ylim(0.0, 7.9); ax.axis("off")
    ax.set_title("The network", fontsize=12, loc="left")
    plt.tight_layout()
    fig.savefig(OUT / "network.png", dpi=150, bbox_inches="tight")


def recipe_search():
    # The three steps of the recipe search, run on each of the three best networks of the depth experiment.
    fig, ax = plt.subplots(figsize=(14, 5.0))
    dot = STEP["training"]

    def header(x, title, sub):
        ax.text(x, 4.75, title, fontsize=11.5, fontweight="bold", color=INK)
        ax.text(x, 4.45, sub, fontsize=9, color=MUTED, va="top")

    def handoff(x0, x1, y, text):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.3))
        ax.text((x0 + x1) / 2, y + 0.12, text, ha="center", va="bottom", fontsize=8.5, color=MUTED, style="italic")

    # 1. grid search: learning rate x weight decay
    header(0.0, "1   learning rate × weight decay", "grid search, 12 runs per network")
    wds = ["$10^{-4}$", r"$5{\cdot}10^{-4}$", "$10^{-3}$", r"$2{\cdot}10^{-3}$", r"$5{\cdot}10^{-3}$", "$10^{-2}$"]
    for i, wd in enumerate(wds):
        x = 0.75 + 0.62 * i
        ax.text(x, 1.95, wd, ha="center", va="top", fontsize=8.5, color=MUTED)
        for y in (2.55, 3.15):
            ax.plot(x, y, "o", color=dot, markersize=9)
    for y, lr in ((3.15, "0.1"), (2.55, "0.05")):
        ax.text(0.4, y, lr, ha="right", va="center", fontsize=8.5, color=MUTED)
    ax.text(0.75 + 0.62 * 2.5, 1.55, "weight decay", ha="center", va="top", fontsize=9, color=INK)
    ax.text(-0.15, 2.85, "learning\nrate", ha="right", va="center", fontsize=9, color=INK)
    ax.text(0.0, 0.85, "other parameters fixed: label smoothing 0.1,\ndropout 0.2, random erasing 0.25,\n"
            "minimum crop 50%, cosine schedule", fontsize=8.5, color=MUTED, va="top")
    handoff(4.2, 5.4, 2.85, "best learning rate\nand weight decay")

    # 2. random search: the same 12 configurations on every network (read from the stored runs)
    x2 = 5.6
    header(x2, "2   random search", "12 configurations drawn once, the same for every network: 12 runs per network")
    configs = [load("tune_random", f"plain_2222_cfg{i:02d}")["config"] for i in range(12)]
    params = [("label\nsmoothing", "label_smoothing", [0.0, 0.05, 0.1, 0.2], ["0", "0.05", "0.1", "0.2"]),
              ("dropout", "dropout", [0.0, 0.2, 0.5], ["0", "0.2", "0.5"]),
              ("random\nerasing", "random_erasing", [0.0, 0.25, 0.5], ["0", "0.25", "0.5"]),
              ("minimum\ncrop", "crop_scale", [0.25, 0.35, 0.5], ["25%", "35%", "50%"]),
              ("schedule", "schedule", ["cosine", "onecycle"], ["cosine", "one-cycle"])]
    row_h, col_w, group_gap, top = 0.19, 0.3, 0.3, 3.55
    x = x2 + 1.55
    for title, key, values, texts in params:
        x_first = x
        for v, t in zip(values, texts):
            ax.text(x, top - 12 * row_h - 0.02, t, ha="center", va="top", fontsize=7.5, color=MUTED, rotation=90)
            for r, c in enumerate(configs):
                chosen = c[key][0] if key == "crop_scale" else c.get(key, "cosine")
                y = top - r * row_h
                ax.plot(x, y, "o", color=dot if chosen == v else "#e4e2da", markersize=5.5 if chosen == v else 3)
            x += col_w
        ax.text((x_first + x - col_w) / 2, top + 0.2, title, ha="center", va="bottom", fontsize=8.5, color=INK)
        x += group_gap
    for r in range(12):
        ax.text(x2 + 1.3, top - r * row_h, f"configuration {r}", ha="right", va="center", fontsize=7.5,
                color=MUTED)
    handoff(x - 0.05, x + 1.05, 2.85, "two best\nconfigurations")

    # 3. confirmation with two more seeds
    x3 = x + 1.3
    header(x3, "3   confirmation", "4 more runs per network")
    for k, seed in enumerate((42, 43, 44)):
        xs = x3 + 1.3 + 0.55 * k
        ax.text(xs, 2.25, str(seed), ha="center", va="top", fontsize=8.5, color=MUTED)
        for y in (2.55, 3.15):
            ax.plot(xs, y, "o", markersize=9, color=dot, markerfacecolor="white" if seed == 42 else dot,
                    markeredgewidth=1.5)
    ax.text(x3 + 1.85, 1.9, "seed", ha="center", va="top", fontsize=9, color=INK)
    for y, t in ((3.15, "best"), (2.55, "second")):
        ax.text(x3 + 1.0, y, t, ha="right", va="center", fontsize=8.5, color=MUTED)
    ax.text(x3, 0.85, "hollow: the run of step 2, reused\nselection on the mean over the seeds; then\n"
            "the best network, or the simpler one if\nthe best two are within one standard deviation",
            fontsize=8.5, color=MUTED, va="top")
    ax.set_xlim(-0.9, x3 + 3.3); ax.set_ylim(-0.1, 5.0); ax.axis("off")
    ax.set_title("The same three steps for each of the three networks: 28 runs per network", fontsize=12,
                 loc="left")
    plt.tight_layout()
    fig.savefig(OUT / "recipe_search.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    roadmap()
    depth_grid()
    data_pipeline()
    network()
    recipe_search()
    print("written:", *(OUT / f for f in ("roadmap.png", "depth_grid.png", "data_pipeline.png", "network.png", "recipe_search.png")))

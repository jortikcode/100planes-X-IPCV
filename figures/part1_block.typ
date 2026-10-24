// Residual block of the final network of Part 1 (two 3x3 convolutions + skip connection).
// Build: typst compile --format png --ppi 300 figures/part1_block.typ figures/part1_block.png
#import "@preview/neural-netz:0.3.0": draw-network

#set page(width: auto, height: auto, margin: 6mm, fill: white)
#set text(font: "Libertinus Serif")

#let layers = (
  (
    type: "custom", name: "x", widths: (0.4,), height: 5, depth: 5,
    fill: rgb("#9aa5b1"), label: "input x", channels: ("C", "H"),
  ),
  (
    type: "conv", name: "c1", widths: (0.5,), height: 5, depth: 5,
    label: "3×3 conv, BN, ReLU", offset: 3.4, show-relu: true,
  ),
  (
    type: "conv", name: "c2", widths: (0.5,), height: 5, depth: 5,
    label: "3×3 conv, BN", offset: 3.4, show-relu: false,
  ),
  (type: "sum", name: "add", label: "add", offset: 3.4),
  (
    type: "custom", name: "y", widths: (0.4,), height: 5, depth: 5,
    fill: rgb("#9aa5b1"), bandfill: rgb("#c0392b"), show-relu: true,
    label: "ReLU, output y", offset: 2.4,
  ),
)

#draw-network(
  layers,
  connections: (
    (from: "x", to: "add", type: "skip", mode: "air", pos: 6.2,
     label: "identity, or 1×1 conv + BN when the stride is 2 or the channels change"),
  ),
  show-legend: false,
  scale: 90%,
)

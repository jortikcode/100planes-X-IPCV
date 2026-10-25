// Final network of Part 1 (res_3461): residual network with 31 convolutional layers.
// Build: typst compile --format png --ppi 300 figures/part1_network.typ figures/part1_network.png
#import "@preview/neural-netz:0.3.0": draw-network

#set page(width: auto, height: auto, margin: 6mm, fill: white)
#set text(font: "Libertinus Serif")

// the height and depth of a layer follow the spatial resolution of its feature map
#let layers = (
  (
    type: "input", image: image("aircraft_example_input.png"), height: 10, depth: 10,
    label: "input", channels: (3, 224),
  ),
  (
    type: "conv", widths: (0.3, 0.3, 0.3), height: 8, depth: 8,
    label: "stem", channels: (32, 32, 64, 112), offset: 3.2,
  ),
  (type: "pool", height: 6.2, depth: 6.2),
  (
    type: "convres", widths: (0.45, 0.45, 0.45), height: 6.2, depth: 6.2,
    label: "stage 1: 3 blocks", channels: (64, 64, 64, 56), offset: 3.6,
  ),
  (
    type: "convres", widths: (0.55, 0.55, 0.55, 0.55), height: 4.6, depth: 4.6,
    label: "stage 2: 4 blocks", channels: (128, 128, 128, 128, 28), offset: 3.2,
  ),
  (
    type: "convres", widths: (0.65, 0.65, 0.65, 0.65, 0.65, 0.65), height: 3.2, depth: 3.2,
    label: "stage 3: 6 blocks", channels: (256, 256, 256, 256, 256, 256, 14), offset: 3.2,
  ),
  (
    type: "convres", widths: (0.8,), height: 2.2, depth: 2.2,
    label: "stage 4: 1 block", channels: (512, 7), offset: 3.2,
  ),
  (
    type: "gap", label: "global avg pool", height: 1.5, depth: 1.5, offset: 2.6,
  ),
  (
    type: "fc", label: "linear", channels: (100,), height: 3, depth: 0.4, offset: 2.2,
  ),
)

#draw-network(layers, show-relu: true, show-legend: true, legend-title: "Layers", scale: 80%)

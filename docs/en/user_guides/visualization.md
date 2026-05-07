# Visualization

BaoIAD provides visualization tools for anomaly detection results: an automatic hook for test-time visualization and standalone scripts for generating paper-quality figures.

## ADVisualizationHook

The [`ADVisualizationHook`](../../../baoiad/visualization/ad_visualization_hook.py) automatically saves composite visualizations during the test stage. It is registered in the default runtime config ([`configs/_base_/default_runtime.py`](../../../configs/_base_/default_runtime.py)) but **disabled by default**:

```python
# configs/_base_/default_runtime.py
default_hooks = dict(
    ...
    visualization=dict(type='ADVisualizationHook', enable=False),
)
```

### Enabling Visualization

**Option 1**: Enable in your config:

```python
default_hooks = dict(
    visualization=dict(type='ADVisualizationHook', enable=True),
)
```

**Option 2**: Enable via command-line override:

```bash
python tools/test.py <config> <checkpoint> \
    --cfg-options default_hooks.visualization.enable=True
```

### Hook Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `bool` | `False` | Master switch. No-op unless `True`. |
| `interval` | `int` | `1` | Visualize every N batches. |
| `max_samples` | `int` | `0` | Maximum images to save. `0` means unlimited. |
| `show_gt` | `bool` | `True` | Include ground truth mask in the output. |
| `score_threshold` | `float` | `0.5` | Threshold for the binary prediction panel. |
| `save_dir` | `str` | `None` | Output directory. Defaults to `<work_dir>/vis/`. |

### Output Format

Each saved image is a composite PNG with four panels:

```
[Title bar: Score | GT label | ANOMALY/NORMAL judgement]
[Original] [GT Mask] [Anomaly Heatmap] [Thresholded Prediction]
[                  Jet Colorbar                           ]
```

- **Original**: The input image.
- **GT Mask**: Ground truth defect mask (red regions). Shows a "No GT" label if unavailable.
- **Anomaly Heatmap**: Jet-colormap overlay of the anomaly score map on the image, with ground truth contours outlined in blue.
- **Thresholded Prediction**: Binary prediction mask at `score_threshold`, anomaly regions in red.

Files are named by the source image filename (e.g. `000.png`) or by sequential index (`000000.png`) if the filename is unavailable.

## ADVisualizer

The [`ADVisualizer`](../../../baoiad/visualization/ad_visualizer.py) provides the rendering engine used by the hook. It can also be used directly in Python:

```python
from baoiad.visualization.ad_visualizer import ADVisualizer

vis = ADVisualizer(name='demo')

# Single sample
vis.save_result(
    'output.png',
    image,           # (H, W, 3) uint8 BGR
    anomaly_map,     # (H, W) float
    gt_mask=gt,      # (H, W) binary, optional
    pred_score=0.87,
    gt_label=1,
    threshold=0.5,
)

# Batch grid
grid = vis.draw_batch(
    images,          # list of (H, W, 3)
    anomaly_maps,    # list of (H, W)
    gt_masks=gt_masks,
    pred_scores=scores,
    max_images=8,
    ncols=4,
)
```

The visualizer uses the standard MMEngine `Visualizer` interface and stores results via the `LocalVisBackend`.

## Standalone Visualization Scripts

Three scripts in `scripts/` generate paper-quality figures. These are **not** CLI tools with argparse — they are run as Python scripts and configured by editing their `__main__` blocks or calling their functions programmatically.

### scripts/gen_vis_examples.py

Generates a paper-style figure with synthetic anomaly maps for README illustrations. Uses random noise to simulate anomaly scores (no model inference).

```bash
python scripts/gen_vis_examples.py
```

The script calls two functions:

- **`generate_paper_figure(data_root, out_path)`**: Creates a multi-row figure with three MVTec AD categories (bottle, hazelnut, tile). Each row shows Image, GT Mask, Anomaly Heatmap, and Prediction panels.
- **`generate_normal_figure(data_root, out_path)`**: Creates a two-panel figure showing a normal sample with its low anomaly scores.

**Configuration**: Edit the `__main__` block to change `data_root` and `out_dir`. Defaults to `data/mvtec_ad` input and `resources/vis_examples/` output.

### scripts/gen_vis_multi_model.py

Generates a comparison figure across multiple models (PatchCore, PaDiM, SPADE) and categories. Runs real model inference.

```bash
python scripts/gen_vis_multi_model.py
```

The main function is `generate_figure(data_root, out_path, device)`. It:
1. Builds each model and trains on per-category normal images.
2. Runs inference on anomalous test images.
3. Renders a grid: rows = models, columns = categories, cells = [Image, GT, Anomaly Map, Prediction].

**Configuration**: Edit the `__main__` block to change `data_root`, `out_path`, and `device`.

### scripts/gen_vis_real.py

Generates a single-model figure using real PatchCore inference results. More detailed than `gen_vis_examples.py` and shows actual model outputs.

```bash
python scripts/gen_vis_real.py
```

The main function is `generate_figure(data_root, out_path, categories, device)` where `categories` is a list of `(category, defect, image_index)` tuples. It:
1. Builds a PatchCore model with WideResNet-50 backbone.
2. For each category, extracts training features and builds the memory bank.
3. Runs inference and renders the result panels.

**Configuration**: Edit the `__main__` block to change paths, categories, and device.

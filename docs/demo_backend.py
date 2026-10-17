"""Matplotlib backend that exports interactive demo figures as HTML."""

import os
import json
import shutil
import sys
from pathlib import Path

import mpld3
import numpy as np
from matplotlib.patches import Circle
import matplotlib.patheffects as path_effects
from matplotlib.backend_bases import FigureManagerBase, _Backend
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib._pylab_helpers import Gcf


def _mpld3_text(text):
    """Prepare simple MathText labels for SVG rendering in the browser."""
    return text.replace("$", "").replace(r"\eta", "η").replace(r"\\", "").strip()


def _rasterize_large_collections(figure, threshold=1000):
    """Convert dense field meshes to images before mpld3 serializes them."""
    field_layers = []
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    height, width = (int(value) for value in renderer.get_canvas_width_height()[::-1])

    for axis_index, axes in enumerate(figure.axes):
        for patch in axes.patches:
            if isinstance(patch, Circle):
                patch.set_edgecolor("white")
                patch.set_linewidth(max(patch.get_linewidth(), 1.5))
                patch.set_zorder(10)
                patch.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=3, foreground="black"),
                        path_effects.Normal(),
                    ]
                )
        for artist in axes.get_children():
            if hasattr(artist, "get_text") and hasattr(artist, "set_text"):
                text = artist.get_text()
                if text:
                    artist.set_text(_mpld3_text(text))
        for collection in list(axes.collections):
            if len(collection.get_paths()) <= threshold:
                continue

            states = []
            for child_axes in figure.axes:
                for artist in child_axes.get_children():
                    states.append((artist, artist.get_visible()))
                    artist.set_visible(artist is collection or artist is child_axes.patch)

            figure.canvas.draw()
            pixels = np.asarray(figure.canvas.buffer_rgba())
            x0, y0, x1, y1 = collection.axes.bbox.extents
            left = max(0, int(round(float(x0))))
            right = min(width, int(round(float(x1))))
            top = max(0, height - int(round(float(y1))))
            bottom = min(height, height - int(round(float(y0))))
            raster = pixels[top:bottom, left:right].copy()

            for artist, visible in states:
                artist.set_visible(visible)

            x_limits = axes.get_xlim()
            y_limits = axes.get_ylim()
            z_order = collection.get_zorder()
            values = np.ma.filled(collection.get_array(), np.nan).astype(float)
            grid_size = int(round(np.sqrt(values.size)))
            if grid_size * grid_size == values.size:
                field_layers.append(
                    {
                        "axisIndex": axis_index,
                        "xlim": x_limits,
                        "ylim": y_limits,
                        "size": grid_size,
                        "values": values.reshape(grid_size, grid_size).tolist(),
                        "xlabel": _mpld3_text(axes.get_xlabel()),
                        "ylabel": _mpld3_text(axes.get_ylabel()),
                        "title": _mpld3_text(axes.get_title()) or "Field",
                    }
                )
            collection.remove()
            axes.imshow(
                raster,
                extent=(x_limits[0], x_limits[1], y_limits[0], y_limits[1]),
                origin="upper",
                interpolation="nearest",
                aspect="auto",
                zorder=z_order,
            )
            axes.set_xlim(x_limits)
            axes.set_ylim(y_limits)
    return field_layers


def _save_interactive_html(figure, destination, field_layers):
    """Export a responsive mpld3 chart with local JavaScript dependencies."""
    js_source = Path(mpld3.__file__).parent / "js"
    mpld3_js = f"mpld3.v{mpld3.__version__}.min.js"
    for filename in ("d3.v5.min.js", mpld3_js):
        target = destination.parent / filename
        if not target.exists():
            shutil.copyfile(js_source / filename, target)

    renderer = figure.canvas.get_renderer()
    width, height = (int(value) for value in renderer.get_canvas_width_height())
    figid = f"mpld3_{destination.stem.replace('-', '_')}"
    mpld3.save_html(
        figure,
        str(destination),
        figid=figid,
        template_type="simple",
        d3_url="d3.v5.min.js",
        mpld3_url=mpld3_js,
    )

    html = destination.read_text(encoding="utf-8")
    field_layers_json = json.dumps(field_layers, separators=(",", ":"))
    html += f"""
<style>
html, body, #{figid} {{ width: 100%; height: 100%; margin: 0; overflow: hidden; }}
#{figid} svg {{ display: block; width: 100%; height: 100%; }}
#{figid}_tooltip {{ display: none; position: fixed; z-index: 10; pointer-events: none;
  padding: 6px 9px; border-radius: 3px; background: rgba(16, 20, 17, 0.94);
  color: #e8f1ea; font: 12px/1.4 sans-serif; white-space: nowrap; }}
</style>
<div id="{figid}_tooltip"></div>
<script>
const chartRoot = document.getElementById("{figid}");
const chart = chartRoot.querySelector("svg");
for (const label of chart.querySelectorAll("text")) {{
  const parts = label.textContent.split(/([A-Za-zη]_[a-z0-9]+)/g);
  if (parts.length === 1) continue;
  label.textContent = "";
  for (const part of parts) {{
    const match = part.match(/^([A-Za-zη])_([a-z0-9]+)$/);
    if (!match) {{ label.append(document.createTextNode(part)); continue; }}
    label.append(document.createTextNode(match[1]));
    const subscript = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
    subscript.setAttribute("baseline-shift", "sub");
    subscript.setAttribute("font-size", "70%");
    subscript.textContent = match[2];
    label.append(subscript);
  }}
}}
chart.setAttribute("viewBox", "0 0 {width} {height}");
chart.setAttribute("preserveAspectRatio", "xMidYMid meet");
const fieldLayers = {field_layers_json};
const mplFigure = mpld3.figures.find((item) => item.figid === "{figid}");
const tooltip = document.getElementById("{figid}_tooltip");
chart.addEventListener("mousemove", (event) => {{
  const point = chart.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  const mouse = point.matrixTransform(chart.getScreenCTM().inverse());
  const layer = fieldLayers.find((item) => {{
    const axes = mplFigure.axes[item.axisIndex];
    const x = mouse.x - axes.position[0];
    const y = mouse.y - axes.position[1];
    return x >= 0 && x <= axes.width && y >= 0 && y <= axes.height;
  }});
  if (!layer) {{ tooltip.style.display = "none"; return; }}

  const axes = mplFigure.axes[layer.axisIndex];
  const localX = mouse.x - axes.position[0];
  const localY = mouse.y - axes.position[1];
  const xv = axes.lastTransform.rescaleX(axes.xdom).invert(localX);
  const yv = axes.lastTransform.rescaleY(axes.ydom).invert(localY);
  const fx = Math.max(0, Math.min(1, (xv - layer.xlim[0]) / (layer.xlim[1] - layer.xlim[0])));
  const fy = Math.max(0, Math.min(1, (yv - layer.ylim[0]) / (layer.ylim[1] - layer.ylim[0])));
  const col = fx * (layer.size - 1);
  const row = fy * (layer.size - 1);
  const c0 = Math.floor(col), c1 = Math.min(c0 + 1, layer.size - 1);
  const r0 = Math.floor(row), r1 = Math.min(r0 + 1, layer.size - 1);
  const tx = col - c0, ty = row - r0;
  const v00 = layer.values[r0][c0], v01 = layer.values[r0][c1];
  const v10 = layer.values[r1][c0], v11 = layer.values[r1][c1];
  const value = (v00 * (1 - tx) + v01 * tx) * (1 - ty)
    + (v10 * (1 - tx) + v11 * tx) * ty;
  tooltip.textContent = `${{layer.title}} · ${{layer.xlabel}}=${{xv.toFixed(3)}} · `
    + `${{layer.ylabel}}=${{yv.toFixed(3)}} · value=${{value.toPrecision(5)}}`;
  tooltip.style.left = `${{event.clientX + 12}}px`;
  tooltip.style.top = `${{event.clientY + 12}}px`;
  tooltip.style.display = "block";
}});
chart.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});
</script>
"""
    destination.write_text(html, encoding="utf-8")


class _SaveBackend(_Backend):
    FigureCanvas = FigureCanvasAgg
    FigureManager = FigureManagerBase

    @staticmethod
    def show(*, block=None):
        """Save interactive figures to ``docs/assets`` or the configured directory."""
        output_dir = Path(os.environ.get("SCATSOL_DEMO_OUTPUT_DIR", "docs/assets"))
        output_dir.mkdir(parents=True, exist_ok=True)
        demo_name = Path(sys.argv[0]).stem if sys.argv else "demo"

        for index, manager in enumerate(Gcf.get_all_fig_managers(), start=1):
            suffix = f"-{index}" if len(Gcf.get_all_fig_managers()) > 1 else ""
            destination = output_dir / f"{demo_name}{suffix}.html"
            figure = manager.canvas.figure
            field_layers = _rasterize_large_collections(figure)
            _save_interactive_html(figure, destination, field_layers)
            print(f"Saved {destination}")


FigureCanvas = _SaveBackend.FigureCanvas
FigureManager = _SaveBackend.FigureManager
show = _SaveBackend.show

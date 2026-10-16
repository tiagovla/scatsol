"""Matplotlib backend that saves figures when a demo calls ``plt.show()``."""

import os
import sys
from pathlib import Path

from matplotlib.backend_bases import FigureManagerBase, _Backend
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib._pylab_helpers import Gcf


class _SaveBackend(_Backend):
    FigureCanvas = FigureCanvasAgg
    FigureManager = FigureManagerBase

    @staticmethod
    def show(*, block=None):
        """Save all open figures to ``docs/assets`` or the configured directory."""
        output_dir = Path(os.environ.get("SCATSOL_DEMO_OUTPUT_DIR", "docs/assets"))
        output_dir.mkdir(parents=True, exist_ok=True)
        demo_name = Path(sys.argv[0]).stem if sys.argv else "demo"

        for index, manager in enumerate(Gcf.get_all_fig_managers(), start=1):
            suffix = f"-{index}" if len(Gcf.get_all_fig_managers()) > 1 else ""
            destination = output_dir / f"{demo_name}{suffix}.png"
            manager.canvas.figure.savefig(destination, dpi=160, bbox_inches="tight")
            print(f"Saved {destination}")


FigureCanvas = _SaveBackend.FigureCanvas
FigureManager = _SaveBackend.FigureManager
show = _SaveBackend.show

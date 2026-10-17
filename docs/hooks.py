"""Generate demo figures before MkDocs collects documentation assets."""

import os
import subprocess
import sys
from pathlib import Path


def on_pre_build(config) -> None:
    """Run each demo with the saving Matplotlib backend."""
    root = Path(config.config_file_path).resolve().parent
    docs = root / "docs"
    env = os.environ.copy()
    env["MPLBACKEND"] = "module://demo_backend"
    env["PYTHONPATH"] = os.pathsep.join([str(docs), str(root), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )

    for demo in sorted((root / "demo").glob("*.py")):
        subprocess.run([sys.executable, str(demo)], cwd=root, env=env, check=True)

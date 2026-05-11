"""Compatibility entrypoint for the proposed TGCN-cVAE+MoE visualizer."""

import os
import runpy
import sys


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    visualizer_path = os.path.join(project_root, "proposed", "TGCN_cVAE_MoE", "visualize.py")
    sys.path.insert(0, os.path.dirname(visualizer_path))
    runpy.run_path(visualizer_path, run_name="__main__")


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fts_diffusion.cli import sample_main


if __name__ == "__main__":
    sample_main()

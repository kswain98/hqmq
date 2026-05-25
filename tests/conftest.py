"""pytest configuration: make repo root importable so `import hqmq` works."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

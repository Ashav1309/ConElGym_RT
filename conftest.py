"""
Root conftest.py — adds the project root to sys.path so that
`from src.xxx import ...` works inside tests without installing the package.
"""

import sys
from pathlib import Path

# Ensure project root is on the path when running pytest from any directory
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# R2 RAG Document Assistant — Application Package

import sys
from pathlib import Path

# Add project root to sys.path so we can import 'needful'
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

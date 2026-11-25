#!/usr/bin/env python3
"""CLI entry point."""

import sys
from pathlib import Path

# Ensure the project root is on the path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from finance_moe.runner import main

if __name__ == "__main__":
    main()

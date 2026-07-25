"""
Single entrypoint for the whole project.

Always run this file from the project root, exactly like this:

    python main.py

No -m flag, no dotted module paths to remember. This file lives at the
project root specifically so Python automatically treats DBA_Agent\ as
the root of all imports (Agents\, Database\, Graphs\, Tools\, etc.),
without you needing to think about how Python resolves packages.
"""

import asyncio
from utils.Logging_Config import setup_logging

setup_logging()

from Agents.DBA_Agent import run_cli

if __name__ == "__main__":
    asyncio.run(run_cli())
import os
import sys
from pathlib import Path

os.environ["LLM_PROVIDER"] = "offline"  # tests never hit the network
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# conftest.py：让 pytest 能 import 到项目根目录下的 core / ui 包。
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

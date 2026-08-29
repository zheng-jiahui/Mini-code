#!/usr/bin/env python
"""
MiniCode 入口。

    python run.py                              # 交互式
    python run.py -t "任务描述"                 # 单次任务
    python run.py --mock -t "离线演示"          # 不联网的自检/演示
    python run.py --print-prompt               # 打印 System Prompt
    python run.py --list-tools                 # 打印工具清单
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

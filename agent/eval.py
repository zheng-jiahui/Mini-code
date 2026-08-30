"""
评测台：用一组标准任务量化 agent 的能力，输出可复现的报告。

    python -m agent.eval                      跑全部任务
    python -m agent.eval --task calc,dedup    只跑指定任务（逗号分隔）
    python -m agent.eval --json               输出机器可读结果，便于两次运行做对比

为什么需要它
------------
单测证明"零件没坏"，评测回答的是另一个问题：**这些零件装起来能不能干成活**。
答辩时最有说服力的从来不是"我有 90 个测试"，而是"我在 10 个标准任务上跑下来，
产物验证通过率 8/10，失败的那两个原因是 X"。后者才是能力证据。

核心设计：以产物为准，不以模型自述为准
--------------------------------------
这是本模块唯一真正重要的决定。判断任务是否完成有两条路：

    A. 问模型 / 看 finish_reason —— 便宜，但**不可信**；
    B. 真的把产物跑起来、喂输入、比对输出 —— 贵，但**是事实**。

选 B。理由是它与本项目已有三处机制同一条原则：假完成拦截（改过文件没验证就
收尾要拦下）、自修复回灌（把 traceback 和源码摆到模型面前而不是听它说"修好了"）、
/diff 可审阅（让用户自己看改了什么）。既然对模型的每一句自述都保留，
那么在"这个 agent 到底行不行"这件事上，更没有理由采信它的自述。

由此得到一个副产品指标：**自述结局与产物验证的偏差**。
· 自述 finish 但验证失败 = **假完成**（最危险的一类，用户拿到跑不起来的代码还以为好了）
· 自述失败但验证通过 = 悲观失败（能力被低估，通常是模型过于保守不敢收尾）
这两类的数量和占比，比单一成功率更能说明问题出在哪。

任务设计的四条约束
------------------
1. **只用标准库**：不引入第三方依赖，避免把"环境装没装对"混进能力评分。
2. **可判定**：每个任务自带 verify()，喂确定输入、比对确定输出，不靠人工看。
3. **每个任务考察一个此前没考察过的维度**：10 个任务 = 10 个维度，而不是
   10 道同一道题。加任务之前先问"它比现有的多测了什么"——如果答不上来，
   那只是把样本量灌大，报告里的通过率会变好看，信息量却没增加。
4. **每个任务的验证器都要过自检**：一份"正确参考解"必须放行、一份
   "看起来像但其实错的解"必须拦下。这是硬要求，见 tests 里的两条测试。
   一个恒为真的验证器不会报错，只会安静地给出 100% 通过率——比没有验证器更危险。

当前 10 个维度（失败在哪一档是有信息量的诊断，例如"只会新建、修不了 bug"）：
    新建+异常边界 / 修 bug / 读数据出报表 / 算法（保序去重） / 数据结构（LRU 淘汰）
    / CLI 参数契约 / 非结构化文本→结构化统计 / 递归嵌套结构 / 跨模块协作与状态一致性
    / 日志抽取与过滤

局限（如实写）
--------------
· 10 个任务样本仍然不大，结论只能说明"在这些任务上"，不能外推成通用能力。
· verify 只能验证接口行为，验证不了代码质量（命名、结构、是否有多余注释）。
· 真实端点有随机性，单次结果可能波动；要看趋势应多次运行取分布。
· 评测不检查"过程是否合理"——靠碰运气跑通的也会得满分。
· 任务全部是"从零写小模块"，不含真实工程里的长上下文改造（读几千行代码后
  改一处），这类能力目前只能靠长任务实测来佐证。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

__all__ = ["EvalTask", "EvalOutcome", "TASKS", "run_suite", "render_report", "main"]

_PY = sys.executable


# ----------------------------------------------------------------------------
# 验证辅助
# ----------------------------------------------------------------------------
def _child_env() -> Dict[str, str]:
    """子进程环境：去掉 PYTHONPATH。

    评测的是"生成的产物能不能跑"，不该被宿主注入的 sitecustomize 之类的东西
    影响结果——两边环境越干净、越独立，结论越可信。
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def _run_py(cwd: Path, code: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """在任务目录里执行一段 Python，返回完成结果。"""
    return subprocess.run(
        [_PY, "-c", code],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
        env=_child_env(),
    )


def _check_import(module: str, assertions: str, timeout: int = 30) -> Callable[[Path], Tuple[bool, str]]:
    """生成一个验证器：导入产物模块并执行断言。

    断言写在子进程里跑，因此产物里哪怕有 import 即崩的写法（语法错误、
    顶层死循环外的异常），也只是这条任务判失败，不会带崩整个评测。
    """
    code = f"import {module}\n{assertions}\nprint('__OK__')\n"

    def _verify(task_dir: Path) -> Tuple[bool, str]:
        try:
            r = _run_py(task_dir, code, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "验证超时（产物可能存在死循环）"
        if r.returncode == 0 and "__OK__" in r.stdout:
            return True, "产物运行通过"
        detail = (r.stderr or r.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"退出码 {r.returncode}"
        return False, f"产物未通过验证：{tail[:200]}"
    return _verify


def _check_stdout(script: str, expects, timeout: int = 30) -> Callable[[Path], Tuple[bool, str]]:
    """生成一个验证器：运行产物脚本，检查其标准输出。

    `expects` 是一组 (说明, 判断函数)，判断函数拿到整段 stdout。
    用函数而不是字符串精确匹配——模型输出格式（"平均分: 85.0" 还是
    "平均分为 85.0"）本就不该成为评分项，我们只关心**算得对不对**。
    """
    def _verify(task_dir: Path) -> Tuple[bool, str]:
        try:
            r = subprocess.run(
                [_PY, script], cwd=str(task_dir), capture_output=True, text=True,
                timeout=timeout, encoding="utf-8", errors="replace", env=_child_env(),
            )
        except subprocess.TimeoutExpired:
            return False, "运行超时"
        if r.returncode != 0:
            tail = (r.stderr or "").strip().splitlines()
            return False, f"脚本退出码 {r.returncode}：{tail[-1][:200] if tail else '无输出'}"
        out = r.stdout or ""
        for label, fn in expects:
            if not fn(out):
                return False, f"输出不符（{label}）：{out.strip()[:200]!r}"
        return True, "输出正确"
    return _verify


def _has_number(target: float, tol: float = 0.05):
    """从整段输出里找是否存在与 target 接近的数字（容差内）。"""
    def _fn(out: str) -> bool:
        for m in re.finditer(r"-?\d+(?:\.\d+)?", out):
            try:
                if abs(float(m.group()) - target) <= tol:
                    return True
            except ValueError:
                continue
        return False
    return _fn


def _run_cli(cwd: Path, script: str, args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """在任务目录里以给定参数运行产物脚本。"""
    return subprocess.run(
        [_PY, script, *args], cwd=str(cwd), capture_output=True, text=True,
        timeout=timeout, encoding="utf-8", errors="replace", env=_child_env(),
    )


def _check_cli(script: str, cases, timeout: int = 30) -> Callable[[Path], Tuple[bool, str]]:
    """生成一个验证器：用**多组**命令行参数运行产物，逐组检查标准输出。

    为什么必须多组参数：只喂一组参数时，"把答案硬编码进脚本"就能通过——
    脚本什么都不解析、直接 `print` 出期望文本，验证器一样满意。
    而 CLI 契约的本质恰恰是**输入变，输出跟着变**；不换输入就测不出这一点。
    换几组参数是最便宜、也最容易实现的防硬编码手段。
    """
    def _verify(task_dir: Path) -> Tuple[bool, str]:
        for args, expects in cases:
            try:
                r = _run_cli(task_dir, script, args, timeout=timeout)
            except subprocess.TimeoutExpired:
                return False, f"运行超时（参数：{' '.join(args)}）"
            if r.returncode != 0:
                tail = (r.stderr or "").strip().splitlines()
                return False, (f"退出码 {r.returncode}（参数：{' '.join(args)}）："
                               f"{tail[-1][:160] if tail else '无输出'}")
            out = r.stdout or ""
            for label, fn in expects:
                if not fn(out):
                    return False, (f"参数 {' '.join(args)} 输出不符（{label}）："
                                   f"{out.strip()[:200]!r}")
        return True, "多组参数下输出均正确"
    return _verify


def _line_numbers(out: str) -> set:
    """抽取输出里所有形如 `12:内容` 的行号（行首数字 + 冒号）。"""
    return {int(m) for m in re.findall(r"(?m)^\s*(\d+)\s*:", out or "")}


def _has_line(out: str, *parts: str) -> bool:
    """输出里存在某一行，同时包含所有给定片段。"""
    return any(all(p in ln for p in parts) for ln in (out or "").splitlines())


def _first_nonempty_line(out: str) -> str:
    for ln in (out or "").splitlines():
        if ln.strip():
            return ln
    return ""


def _pair_line(word: str, count: int):
    """输出中存在某一行**同时**包含该词与该数字。

    用于词频这类"词 → 次数"的配对输出：只要求"同一个词和它的次数出现在同一行"，
    而不规定写成 `apple 4` 还是 `apple: 4` 还是 `apple 出现了 4 次`。
    措辞不该成为评分项，我们只关心**配得对不对**。
    """
    def _fn(out: str) -> bool:
        w = word.lower()
        for line in (out or "").splitlines():
            low = line.lower()
            if w in low and re.search(rf"(?<!\d){count}(?!\d)", low):
                return True
        return False
    return _fn


# ----------------------------------------------------------------------------
# 任务定义
# ----------------------------------------------------------------------------
@dataclass
class EvalTask:
    name: str                                   # 短名，用于 --task 过滤与目录名
    intent: str                                 # 考察点，报告里说明"这题在测什么"
    prompt: str                                 # 发给 agent 的任务描述
    files: Dict[str, str] = field(default_factory=dict)   # 预置到任务目录的文件
    verify: Optional[Callable[[Path], Tuple[bool, str]]] = None


TASKS: List[EvalTask] = [
    EvalTask(
        name="calc",
        intent="新建文件 + 除 0 边界",
        prompt=(
            "在工作区新建 `calc.py`，实现四个纯函数：add(a, b)、sub(a, b)、mul(a, b)、div(a, b)。\n"
            "要求：div 在除数为 0 时必须抛出 ValueError（不要返回 None 或 inf），"
            "其余三个直接返回数值结果。\n"
            "写完后用 run_command 跑一段自测代码，验证四个函数（含除 0 抛异常）都正确。"
        ),
        verify=_check_import("calc", """
assert calc.add(2, 3) == 5, 'add 错'
assert calc.sub(5, 3) == 2, 'sub 错'
assert calc.mul(2, 3) == 6, 'mul 错'
assert abs(calc.div(7, 2) - 3.5) < 1e-9, 'div 错'
try:
    calc.div(1, 0)
    raise AssertionError('除 0 未抛 ValueError')
except ValueError:
    pass
"""),
    ),
    EvalTask(
        name="fixbug",
        intent="读代码 → 定位 → 修 bug",
        prompt=(
            "工作区已有 `stats.py`，其中 average 函数算出来的结果不对："
            "average([1, 2, 3]) 应该得到 2.0，但现在算错了。\n"
            "请定位根因并修好它，然后用 run_command 运行验证，确认 average([1, 2, 3]) == 2.0。"
        ),
        files={"stats.py": (
            "def total(items):\n"
            "    s = 0\n"
            "    for x in items:\n"
            "        s += x\n"
            "    return s\n"
            "\n"
            "\n"
            "def average(items):\n"
            "    if not items:\n"
            "        return 0\n"
            "    return total(items) / (len(items) - 1)\n"
        )},
        verify=_check_import("stats", """
assert abs(stats.average([1, 2, 3]) - 2.0) < 1e-9, 'average([1,2,3]) 应得 2.0'
assert abs(stats.average([2, 4]) - 3.0) < 1e-9, 'average([2,4]) 应得 3.0'
assert stats.average([]) == 0, '空列表应保持返回 0'
assert stats.total([1, 2, 3]) == 6, 'total 不该被改坏'
"""),
    ),
    EvalTask(
        name="csvreport",
        intent="读 CSV → 算指标 → 出报表",
        prompt=(
            "工作区已有 `scores.csv`（含表头 `姓名,分数`，共 4 行数据）。\n"
            "请新建 `report.py`：读取该 CSV，计算所有分数的平均分（保留一位小数）与最高分，"
            "并把结果打印出来，分别以 `平均分:` 和 `最高分:` 开头各占一行。\n"
            "写完用 run_command 运行，确认输出的两个数字与手算一致。"
        ),
        files={"scores.csv": "姓名,分数\n张三,85\n李四,92\n王五,74\n赵六,89\n"},
        # 平均 (85+92+74+89)/4 = 85.0，最高 92
        verify=_check_stdout("report.py", [
            ("平均分应为 85.0", _has_number(85.0, tol=0.05)),
            ("最高分应为 92", _has_number(92.0, tol=0.05)),
        ]),
    ),
    EvalTask(
        name="dedup",
        intent="保序去重 + 不改入参",
        prompt=(
            "新建 `dedup.py`，实现 `dedup(items)`：去掉重复元素，但**保持每个元素首次出现的顺序**，"
            "返回一个新列表，不得修改传入的原列表。\n"
            "写完用 run_command 运行验证：dedup([3, 1, 3, 2, 1]) 应得到 [3, 1, 2]。"
        ),
        verify=_check_import("dedup", """
assert dedup.dedup([3, 1, 3, 2, 1]) == [3, 1, 2], '基本用例错'
assert dedup.dedup([]) == [], '空列表错'
assert dedup.dedup(['a', 'b', 'a']) == ['a', 'b'], '字符串用例错'
src = [1, 1, 2]
dedup.dedup(src)
assert src == [1, 1, 2], '不该修改入参'
"""),
    ),
    EvalTask(
        name="lru",
        intent="LRU 缓存：满时淘汰最久未用",
        prompt=(
            "新建 `lru.py`，实现 `LRUCache` 类：\n"
            "- `LRUCache(capacity)` 构造，capacity 是容量；\n"
            "- `get(key)` 返回对应的 value，不存在返回 -1；\n"
            "- `put(key, value)` 写入或更新；\n"
            "- 容量已满时再 put 新键，要淘汰**最久未使用**的那个键；\n"
            "- get 算一次使用（会刷新该键的新鲜度）。\n"
            "写完用 run_command 运行验证：容量为 2 时，put(1,1)、put(2,2)、get(1)、put(3,3) 之后，"
            "get(2) 应为 -1（2 被淘汰），get(1) 仍为 1。"
        ),
        # ⚠️ 这些断言的顺序就是 LRU 的定义，写错一步就会把正确实现判成失败
        # （本验证器初版就犯过：put(4,4) 之前刚 get(1)，该被淘汰的是 3 而不是 1）。
        # 因此下面每个断言都注明"此刻谁是最久未用的"，便于复核。
        verify=_check_import("lru", """
from lru import LRUCache
c = LRUCache(2)
c.put(1, 1)                 # 使用顺序（旧→新）：1
c.put(2, 2)                 # 顺序：1, 2
assert c.get(1) == 1, 'get 应返回已写入的值'      # 顺序：2, 1
c.put(3, 3)                 # 满，淘汰最久未用的 2
assert c.get(2) == -1, '应淘汰最久未用的键 2'
assert c.get(1) == 1, '刚用过的 1 不该被淘汰'      # 顺序：3, 1
c.put(4, 4)                 # 满，淘汰最久未用的 3（不是 1）
assert c.get(3) == -1, '应淘汰最久未用的键 3'
assert c.get(1) == 1, '1 仍应命中'
assert c.get(4) == 4, '4 应命中'
"""),
    ),

    # ---- 以下 5 个任务各自补一个此前没考察过的维度 --------------------------
    EvalTask(
        name="findkw",
        intent="CLI 契约：换参数换输出",
        prompt=(
            "新建 `find_kw.py`，作为命令行工具使用：`python find_kw.py <文件路径> <关键字>`。\n"
            "它打印文件中所有包含该关键字的行，每行格式为 `行号:该行原文`（行号从 1 开始，"
            "冒号后直接接原文）。一个都匹配不到时，输出一行 `无匹配`。\n"
            "工作区已有 `notes.txt`（5 行英文）。写完用 run_command 分别以 fix、docs、todo "
            "三个关键字各运行一次，确认行号与命中行数都正确。"
        ),
        files={"notes.txt": (
            "release notes 2026-08\n"
            "fix: login timeout on slow network\n"
            "feat: add export button\n"
            "fix: crash when file is missing\n"
            "docs: update readme\n"
        )},
        # 三组参数是防硬编码的关键：只喂一组的话，"把答案直接 print 出来"就能通过。
        verify=_check_cli("find_kw.py", [
            (["notes.txt", "fix"], [("应命中第 2、4 行", lambda out: _line_numbers(out) == {2, 4})]),
            (["notes.txt", "docs"], [("应命中第 5 行", lambda out: _line_numbers(out) == {5})]),
            (["notes.txt", "todo"], [("无匹配时应明确说明", lambda out: "无匹配" in out)]),
        ]),
    ),
    EvalTask(
        name="wordfreq",
        intent="词频统计（归一 + 排序）",
        prompt=(
            "新建 `wordfreq.py`：读取同目录的 `article.txt`，统计其中出现次数最多的"
            "**前 3 个单词**及其次数并打印。\n"
            "规则：比较时忽略大小写，并忽略紧挨着单词的标点符号——"
            "因此 `Apple,` 与 `apple` 算同一个词。\n"
            "输出按次数从高到低，每行一个，写成 `单词 次数`（例如 `apple 4`），只输出前 3 个。\n"
            "写完用 run_command 运行确认。"
        ),
        files={"article.txt": (
            "apple banana apple cherry\n"
            "Apple, banana cherry\n"
            "apple banana\n"
            "date\n"
        )},
        # 归一化后：apple 4 / banana 3 / cherry 2 / date 1（前 3 名不含 date）
        verify=_check_stdout("wordfreq.py", [
            ("apple 应出现 4 次", _pair_line("apple", 4)),
            ("banana 应出现 3 次", _pair_line("banana", 3)),
            ("cherry 应出现 2 次", _pair_line("cherry", 2)),
            ("只取前 3 个，不该输出 date", lambda out: not _pair_line("date", 1)(out)),
        ]),
    ),
    EvalTask(
        name="flatten",
        intent="递归展平任意深度嵌套",
        prompt=(
            "新建 `flatten.py`，实现 `flatten(items)`：把**任意深度**嵌套的列表展平成一维列表，"
            "保持元素原有顺序，返回一个新列表，不得修改传入的原列表。\n"
            "空列表、或列表里全是空列表，都应返回 `[]`。\n"
            "写完用 run_command 运行验证：flatten([1, [2, [3, 4], 5], [], 6]) 应得到 [1, 2, 3, 4, 5, 6]。"
        ),
        # 关键是"任意深度"：只展平一层的实现会在深嵌套用例上露馅。
        verify=_check_import("flatten", """
assert flatten.flatten([1, [2, [3, 4], 5], [], 6]) == [1, 2, 3, 4, 5, 6], '嵌套展平错'
assert flatten.flatten([]) == [], '空列表错'
assert flatten.flatten([1, 2, 3]) == [1, 2, 3], '本就平坦的列表应原样返回'
assert flatten.flatten([[[[[1]]]]]) == [1], '深嵌套错'
assert flatten.flatten([[], [[]]]) == [], '元素全是空列表应得 []'
src = [1, [2]]
flatten.flatten(src)
assert src == [1, [2]], '不该修改入参'
"""),
    ),
    EvalTask(
        name="bank",
        intent="跨模块 + 转账原子性",
        prompt=(
            "新建两个文件 `account.py` 与 `bank.py`：\n"
            "- `account.py` 定义 `Account` 类：`Account(name, balance=0)`，属性 `.name` / `.balance`，"
            "方法 `deposit(amount)`、`withdraw(amount)`（**余额不足时抛 ValueError，且余额保持不变**）。\n"
            "- `bank.py` 用 `from account import Account` 复用它，定义 `Bank` 类："
            "`open(name, balance=0)` 开账户并返回该账户；`get(name)` 取账户；"
            "`transfer(源名字, 目标名字, amount)` 转账（源账户余额不足时抛 ValueError）；"
            "`total()` 返回所有账户余额之和。\n"
            "关键约束：转账要么完整发生、要么完全不发生——失败的那次转账不能留下改过的余额。\n"
            "写完用 run_command 运行验证（含失败转账后余额未变的检查）。"
        ),
        # 这一题真正测的是"失败操作的原子性"：多数实现只顾成功路径，
        # 失败时会先把源账户扣成负数再抛异常。total() 是抓它的抓手。
        verify=_check_import("bank", """
from account import Account
from bank import Bank
a = Account('a', 100)
a.deposit(50)
assert a.balance == 150, 'deposit 错'
a.withdraw(30)
assert a.balance == 120, 'withdraw 错'
try:
    a.withdraw(999)
    raise AssertionError('余额不足的取款应抛 ValueError')
except ValueError:
    pass
assert a.balance == 120, '失败的取款不该改变余额'

b = Bank()
b.open('x', 100)
b.open('y', 0)
b.transfer('x', 'y', 40)
assert b.get('x').balance == 60, '转出方余额错'
assert b.get('y').balance == 40, '转入方余额错'
assert b.total() == 100, '转账不该改变总余额'
try:
    b.transfer('y', 'x', 999)
    raise AssertionError('余额不足的转账应抛 ValueError')
except ValueError:
    pass
assert b.total() == 100, '失败的转账不该留下改动（缺原子性）'
assert b.get('y').balance == 40, '失败的转账不该扣钱'
"""),
    ),
    EvalTask(
        name="parselog",
        intent="日志过滤：按级别抽取并保序",
        prompt=(
            "工作区已有 `server.log`（6 行日志，格式为 `日期 时间 级别 消息`，级别为 "
            "INFO / WARN / ERROR 之一）。\n"
            "新建 `parselog.py`：读取该文件，挑出所有级别为 ERROR 的行。运行后打印——\n"
            "第一行固定为 `ERROR 条数: N`；随后按**在文件中出现的顺序**每行一条，"
            "格式为 `HH:MM:SS 消息内容`（不要带日期部分）。\n"
            "不要把 WARN / INFO 行算进来。写完用 run_command 运行确认。"
        ),
        files={"server.log": (
            "2026-08-30 10:00:01 INFO  started\n"
            "2026-08-30 10:00:05 ERROR db timeout\n"
            "2026-08-30 10:00:09 WARN  slow query\n"
            "2026-08-30 10:00:12 ERROR auth failed\n"
            "2026-08-30 10:00:20 INFO  retrying\n"
            "2026-08-30 10:00:31 ERROR cache miss\n"
        )},
        verify=_check_stdout("parselog.py", [
            ("第一行应报 ERROR 共 3 条",
             lambda out: bool(re.search(r"(?<!\d)3(?!\d)", _first_nonempty_line(out)))),
            ("应含 10:00:05 db timeout", lambda out: _has_line(out, "10:00:05", "db timeout")),
            ("应含 10:00:12 auth failed", lambda out: _has_line(out, "10:00:12", "auth failed")),
            ("应含 10:00:31 cache miss", lambda out: _has_line(out, "10:00:31", "cache miss")),
            ("不该把 WARN 行算进来", lambda out: "slow query" not in (out or "")),
        ]),
    ),
]


# ----------------------------------------------------------------------------
# 执行
# ----------------------------------------------------------------------------
@dataclass
class EvalOutcome:
    task: str
    intent: str
    finish_reason: str
    steps: int
    tool_calls: int
    tool_errors: int
    elapsed: float
    tokens: int
    verify_ok: bool
    verify_msg: str
    error: str = ""

    @property
    def claimed_ok(self) -> bool:
        """模型自述是否完成。"""
        return self.finish_reason in ("finish", "model_final")


def run_suite(
    agent_cfg,
    profile,
    build_backend_fn,
    tasks: List[EvalTask],
    workspace: Path,
    *,
    max_steps: Optional[int] = None,
    on_progress=None,
) -> List[EvalOutcome]:
    """依次跑完所有任务，返回逐任务结果。

    每个任务用**全新的 AgentLoop**：任务之间不能共享历史，否则前一个任务的
    上下文会帮到（或干扰到）后一个，测出来的就不是单任务能力了。
    """
    from .loop import AgentLoop
    from .tools import build_default_registry

    outcomes: List[EvalOutcome] = []
    for t in tasks:
        task_dir = workspace / t.name
        task_dir.mkdir(parents=True, exist_ok=True)
        for rel, content in t.files.items():
            (task_dir / rel).write_text(content, encoding="utf-8")

        cfg = agent_cfg
        if max_steps:
            cfg = _with_max_steps(agent_cfg, max_steps)
        cfg = _with_workspace(cfg, str(task_dir))

        backend = build_backend_fn(profile)
        loop = AgentLoop(cfg, profile, backend, build_default_registry(), console=None)

        started = time.time()
        try:
            result = loop.run(t.prompt)
            err = result.error_message
        except Exception as exc:  # noqa: BLE001 —— 一个任务崩了不该带走整场评测
            result = None
            err = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - started

        rec = loop.metrics.tasks[-1] if loop.metrics.tasks else None
        if result is None:
            outcomes.append(EvalOutcome(
                t.name, t.intent, "internal_error", 0, 0, 0, elapsed, 0,
                False, "任务执行异常", error=err,
            ))
        else:
            ok, msg = (False, "无验证器")
            if t.verify:
                try:
                    ok, msg = t.verify(task_dir)
                except Exception as exc:  # noqa: BLE001
                    ok, msg = False, f"验证器异常：{type(exc).__name__}: {exc}"
            outcomes.append(EvalOutcome(
                t.name, t.intent, result.finish_reason,
                rec.steps if rec else result.steps,
                rec.tool_calls if rec else result.tool_calls,
                rec.tool_errors if rec else result.errors,
                elapsed,
                rec.tokens if rec else 0,
                ok, msg, error=err,
            ))

        if on_progress:
            on_progress(outcomes[-1])
    return outcomes


def _with_workspace(cfg, workspace: str):
    """拷一份配置并换掉工作区（每个评测任务独立目录，不能共用）。

    用 object.__setattr__ 而不是 c.x = v：AgentConfig 是数据类，直接赋值在
    它被冻结时会失败，而 object.__setattr__ 能绕过这一层。
    """
    c = copy.copy(cfg)
    object.__setattr__(c, "workspace", workspace)
    # 产物直接落在任务目录，避免多套一层子目录（预置文件与产物要平级才好用）
    object.__setattr__(c, "per_task_dir", False)
    return c


def _with_max_steps(cfg, max_steps: int):
    c = copy.copy(cfg)
    object.__setattr__(c, "max_steps", int(max_steps))
    return c


# ----------------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------------
def _disp_width(s: str) -> int:
    """字符串的终端显示宽度（CJK 与全角标点占 2 列）。

    报告是要贴进答辩材料的，`{s:<26}` 按**字符数**补齐，而 CJK 在终端里占两列，
    于是含中文的那一列会集体错位——表格糊了，数字就没法读。
    按 East Asian Width 规则算宽度再手工补空格，才能对齐。
    """
    w = 0
    for ch in s:
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _pad(s: str, width: int, align: str = "<") -> str:
    """按显示宽度补齐（超宽时截断，截断也按显示宽度算）。"""
    out, w = "", 0
    for ch in s:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > width:
            break
        out += ch
        w += cw
    gap = " " * max(0, width - w)
    return (out + gap) if align == "<" else (gap + out)


def render_report(outcomes: List[EvalOutcome], *, model: str = "", elapsed: float = 0.0) -> str:
    n = len(outcomes)
    passed = sum(1 for o in outcomes if o.verify_ok)
    claimed = sum(1 for o in outcomes if o.claimed_ok)
    false_done = [o for o in outcomes if o.claimed_ok and not o.verify_ok]
    pessimistic = [o for o in outcomes if not o.claimed_ok and o.verify_ok]

    L: List[str] = []
    L.append("=" * 78)
    L.append("MiniCode 评测报告")
    L.append("=" * 78)
    L.append(f"  模型：{model or '-'}")
    L.append(f"  任务数：{n}    总耗时：{elapsed:.1f}s    生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")
    L.append(f"  产物验证通过：{passed}/{n}"
             + (f"（{passed / n * 100:.0f}%）" if n else ""))
    L.append(f"  模型自述完成：{claimed}/{n}"
             + (f"（{claimed / n * 100:.0f}%）" if n else ""))
    L.append("")
    L.append("  判据：**以产物为准**。上两行的差值就是自述与事实的偏差——")
    L.append("  自述完成但验证失败 = 假完成（用户会拿到跑不起来的代码还以为好了）；")
    L.append("  自述失败但验证通过 = 悲观失败（能力被低估，通常是模型不敢收尾）。")
    L.append("")

    L.append("  " + "-" * 76)
    L.append("  " + _pad("任务", 10) + _pad("考察点", 30) + _pad("结局", 14)
             + _pad("步", 3, ">") + _pad("调用", 5, ">") + _pad("耗时", 8, ">")
             + _pad("产物", 6, ">"))
    L.append("  " + "-" * 76)
    for o in outcomes:
        L.append(
            "  " + _pad(o.task, 10) + _pad(o.intent, 30) + _pad(o.finish_reason, 14)
            + _pad(str(o.steps), 3, ">") + _pad(str(o.tool_calls), 5, ">")
            + _pad(f"{o.elapsed:.1f}s", 8, ">")
            + _pad("✓" if o.verify_ok else "✗", 6, ">")
        )
    L.append("  " + "-" * 76)

    if n:
        L.append("")
        L.append(f"  平均步数 {sum(o.steps for o in outcomes) / n:.1f}    "
                 f"平均工具调用 {sum(o.tool_calls for o in outcomes) / n:.1f}    "
                 f"平均耗时 {sum(o.elapsed for o in outcomes) / n:.1f}s    "
                 f"平均 token {sum(o.tokens for o in outcomes) / n:,.0f}")
        total_calls = sum(o.tool_calls for o in outcomes)
        total_err = sum(o.tool_errors for o in outcomes)
        if total_calls:
            L.append(f"  工具失败率 {total_err / total_calls * 100:.1f}%（{total_err}/{total_calls}）")

    L.append("")
    if false_done:
        L.append(f"  ⚠ 假完成 {len(false_done)} 个（自述完成但产物跑不通）：")
        for o in false_done:
            L.append(f"      · {o.task}：{o.verify_msg}")
    if pessimistic:
        L.append(f"  ⚠ 悲观失败 {len(pessimistic)} 个（产物其实通过了却没敢收尾）：")
        for o in pessimistic:
            L.append(f"      · {o.task}：{o.finish_reason}")
    failed = [o for o in outcomes if not o.verify_ok and not o.claimed_ok]
    if failed:
        L.append(f"  未通过 {len(failed)} 个：")
        for o in failed:
            L.append(f"      · {o.task}（{o.finish_reason}）：{o.verify_msg}")
            if o.error:
                L.append(f"        错误：{o.error[:160]}")
    if not (false_done or pessimistic or failed):
        L.append("  全部任务产物验证通过。")
    L.append("=" * 78)
    return "\n".join(L)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m agent.eval",
        description="MiniCode 评测台：跑一组标准任务并以产物验证为准给出报告",
    )
    p.add_argument("--config", help="配置文件路径（默认 config.yaml）")
    p.add_argument("--profile", help="使用配置里的哪个 LLM profile")
    p.add_argument("--task", help="只跑指定任务，逗号分隔（可选值：" + ",".join(t.name for t in TASKS) + "）")
    p.add_argument("--workspace", help="评测工作区（默认 workplace/_eval/<时间戳>）")
    p.add_argument("--max-steps", type=int, help="覆盖单任务步数上限")
    p.add_argument("--json", action="store_true", help="额外输出 JSON 结果，便于两次运行对比")
    p.add_argument("--quiet", action="store_true", help="不打印逐任务进度")
    args = p.parse_args(argv)

    from .config import load_config
    from .llm import build_backend

    try:
        cfg = load_config(explicit=args.config, profile=args.profile, require_api_key=True)
    except Exception as exc:  # noqa: BLE001
        print(f"配置加载失败：{exc}")
        return 2

    tasks = TASKS
    if args.task:
        wanted = {s.strip() for s in args.task.split(",") if s.strip()}
        unknown = wanted - {t.name for t in TASKS}
        if unknown:
            print(f"未知任务：{', '.join(sorted(unknown))}；可选：{', '.join(t.name for t in TASKS)}")
            return 2
        tasks = [t for t in TASKS if t.name in wanted]

    ws = Path(args.workspace) if args.workspace else (
        Path("workplace") / f"_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    ws = ws.resolve()
    ws.mkdir(parents=True, exist_ok=True)

    def _backend(profile):
        return build_backend(profile)

    def _progress(o: EvalOutcome):
        if not args.quiet:
            mark = "✓" if o.verify_ok else "✗"
            print(f"  [{mark}] {o.task:<10} {o.steps:>2}步 {o.tool_calls:>2}调用 "
                  f"{o.elapsed:>6.1f}s  {o.finish_reason:<14} {o.verify_msg[:60]}")

    print(f"\n评测开始：{len(tasks)} 个任务，模型 {cfg.llm.model}，工作区 {ws}\n")
    started = time.time()
    outcomes = run_suite(
        cfg.agent, cfg.llm, _backend, tasks, ws,
        max_steps=args.max_steps, on_progress=_progress,
    )
    total = time.time() - started

    print()
    print(render_report(outcomes, model=cfg.llm.model, elapsed=total))

    if args.json:
        payload = {
            "model": cfg.llm.model,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": round(total, 2),
            "verify_passed": sum(1 for o in outcomes if o.verify_ok),
            "total": len(outcomes),
            "tasks": [o.__dict__ for o in outcomes],
        }
        out = ws / "eval-result.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON 结果已写入：{out}")

    # 退出码反映"产物验证"而非"模型自述"：全通过才是 0
    return 0 if all(o.verify_ok for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())

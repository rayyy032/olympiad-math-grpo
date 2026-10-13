#!/usr/bin/env python3
"""Generate ``kaggle_train.ipynb`` for the olympiad-math-grpo project.

Emits a self-contained Kaggle T4 notebook that clones the repo, verifies the
27 unit tests, runs a VRAM smoke test, then executes the seven single-variable
ablations with the anti-hang guards proven on the previous project (swanlab
disabled, stdin=DEVNULL, stderr pumped on a thread, watchdog + per-run budget).

Run:  python scripts/build_kaggle_notebook.py
"""

import json
import os

REPO = "https://github.com/rayyy032/olympiad-math-grpo.git"
REPO_DIR = "/kaggle/working/olympiad-math-grpo"
REPO_NAME = "olympiad-math-grpo"

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src})


md(r"""# Olympiad-Math GRPO/DAPO — 竞赛级数学推理的失效机制消融（Kaggle T4）

从零手写 GRPO + DAPO，在 **Qwen2.5-1.5B-Instruct** 上做**竞赛级**数学推理 RL：

| 用途 | 数据集 | 规模 |
|---|---|---|
| 训练 | `math_l45` = MATH level 4-5 | 3994 题（竞赛难度，不是小学 GSM8K） |
| 分布内评测 | `math_l45` test split | 2538 题 |
| **分布外压力测试** | `olympiad` = AIME(90) + AMC(83) + MATH-L5(721) | 894 题（Kaggle AIMO 同源） |

**核心问题**：题目难度从小学跃升到竞赛级后，DAPO 的哪些组件从"可选项"变成"必需品"？
RL 学到的是推理能力，还是 MATH 的数据分布？

**运行方式**：Settings → Accelerator = `GPU T4 x1`，Internet = `ON` → Save Version → **Save & Run All**。

> 显存提示：1.5B + LoRA + group_size=8 + 384 token 在 16G T4 上偏紧。
> 若 Cell 4 冒烟 OOM：把 Cell 6 的 `MAX_NEW_TOKENS` 降到 256、`GROUP_SIZE` 降到 4，或 `BASE_MODEL` 回退 `Qwen/Qwen2.5-0.5B-Instruct`。
""")

code(r"""# Cell 1: GPU 检查
!nvidia-smi
import torch
assert torch.cuda.is_available(), "请在右侧 Settings -> Accelerator 选择 GPU T4 后重新运行"
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)
print("VRAM(G):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
""")

code(r"""# Cell 2: clone 仓库 + 安装依赖 + 验证文件完整性
import os, subprocess, sys

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["PYTHONUNBUFFERED"] = "1"

print("=== git clone ===")
result = subprocess.run(["git", "clone", "https://github.com/rayyy032/olympiad-math-grpo.git"],
                        capture_output=True, text=True)
print("stdout:", result.stdout)
if result.stderr:
    print("stderr:", result.stderr)
print("returncode:", result.returncode)
assert result.returncode == 0, f"git clone 失败: {result.stderr}"

%cd /kaggle/working/olympiad-math-grpo

critical_files = [
    "scripts/train.py", "scripts/run_ablations.sh",
    "src/config.py", "src/rewards.py", "src/losses.py",
    "src/generation.py", "src/data.py", "src/evaluation.py",
    "src/diagnostics.py", "src/train_utils.py",
    "src/agents/base_agent.py", "src/agents/grpo_agent.py",
    "tests/test_reward.py",
]
print("\n=== 检查关键文件 ===")
all_exist = True
for f in critical_files:
    exists = os.path.isfile(f)
    size = os.path.getsize(f) if exists else 0
    status = "OK" if exists and size > 0 else "MISSING/EMPTY"
    print(f"  [{status}] {f} ({size} bytes)")
    if not exists or size == 0:
        all_exist = False
assert all_exist, "有文件缺失！检查 git clone 是否成功"

print("\n=== pip install ===")
result = subprocess.run([sys.executable, "-m", "pip", "install",
    "transformers>=4.45", "datasets", "peft", "accelerate",
    "math-verify", "swanlab", "pyyaml", "tqdm", "matplotlib"],
    capture_output=True, text=True)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.stderr:
    print("stderr:", result.stderr[-1500:])
print("returncode:", result.returncode)

# torchao: Kaggle 预装 0.10.0，PEFT 要求 >0.16.0，不卸载会导致 LoRA 包装 ImportError
print("\n=== 卸载 torchao ===")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               capture_output=True, text=True)
print("torchao uninstalled (if it was installed)")

print("\n=== 验证 import ===")
for mod in ["torch", "transformers", "peft", "datasets", "numpy", "yaml", "matplotlib"]:
    try:
        m = __import__(mod)
        print(f"  [OK] {mod} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  [FAIL] {mod}: {e}")
try:
    import math_verify
    print("  [OK] math_verify")
except Exception as e:
    print(f"  [FAIL] math_verify: {e}  (仅影响数值等价 fallback)")
""")

code(r"""# Cell 3: 27 项边界单测（完整输出）
import subprocess, sys
result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
                        capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print("returncode:", result.returncode)
""")

code(r"""# Cell 4: 显存冒烟 — 用目标配置真跑 1 步，测出峰值显存
# 失败就说明 Cell 6 的配置在 T4 上放不下，按 Cell 0 的提示降参后再继续。
import subprocess, sys, os
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SMOKE_BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # 与 Cell 6 保持一致
proc = subprocess.run(
    [sys.executable, "scripts/train.py",
     "--algorithm", "dapo",
     "--dataset", "math_l45",              # 真数据，验证数据管线
     "--eval-dataset", "aime,math_l45",
     "--base-model", SMOKE_BASE,
     "--train-samples", "8", "--eval-samples", "3", "--max-steps", "1",
     "--group-size", "8", "--groups-per-step", "1",
     "--max-new-tokens", "384", "--max-prompt-tokens", "384",
     "--device", "cuda", "--use-swanlab", "false",
     "--output-dir", "/kaggle/working/_smoke"],
    capture_output=True, text=True, env=env)
print("=== STDOUT ===")
print(proc.stdout[-4000:])
if proc.stderr:
    print("=== STDERR ===")
    print(proc.stderr[-3000:])
print("=== returncode:", proc.returncode, "===")
if proc.returncode != 0:
    print("\n!! 冒烟失败。若为 CUDA out of memory：把 Cell 6 的 MAX_NEW_TOKENS 降到 256、"
          "GROUP_SIZE 降到 4，或 BASE_MODEL 回退 0.5B，然后重跑 Cell 4。")
    raise RuntimeError("Smoke test failed — see traceback above")
print("\nOK 冒烟通过（1.5B + 384 token 在 T4 上放得下），可以跑七组消融")
print("提示：想要峰值显存数字，把上面命令的 --max-steps 改成 2 并加 --run-name smoke 手动看 nvidia-smi")
""")

md(r"""## 七组单变量消融

下面的 cell 依次跑 7 组。每组实时输出每步的 reward / KL / surprisal / **零方差组占比** / 补全长度 / hack rates。

**竞赛题的关键观测量**：`zero_variance_group_ratio`（难题下会显著抬升，是 dynamic sampling 的价值所在）
与 `mean_completion_tokens`（长 CoT，是 overlong shaping 的价值所在）。

每组评测**两个集合**：`[eval:olympiad]`（分布外）与 `[eval:math_l45]`（分布内）。

防挂死（沿用上一版 12h 白烧的教训）：swanlab 关闭、stdin=DEVNULL、stderr 后台线程抽干、
看门狗 10 分钟无输出杀进程、单组预算、全局截止。
快速验证把 `MAX_STEPS` 改成 5。
""")

code(r"""# ============================================================
# Cell 6: 七组单变量消融（唯一需要改的配置区）
# ============================================================
PRESETS = ["grpo", "clip_higher", "dynamic", "token_level",
           "overlong", "dapo", "entropy_reg"]
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET = "math_l45"                 # 训练集：竞赛级 MATH L4-5
EVAL_DATASET = "olympiad,math_l45"   # 分布外 + 分布内
TRAIN_SAMPLES = 3000
EVAL_SAMPLES = 150
MAX_STEPS = 150
GROUP_SIZE = 8
GROUP_PER_STEP = 1
MAX_NEW_TOKENS = 384
PPO_EPOCHS = 4                       # 让 clip 真正生效
LEARNING_RATE = "2e-5"
OVERLONG_CACHE = 128                 # 软惩罚窗口：[-256,384] 区间线性，对齐 DAPO 比例
QUIET_KILL_MIN = 10                  # 连续多少分钟无输出判定挂死
BUDGET_MIN = 130                     # 单组预算
GLOBAL_DEADLINE_H = 10.5
# ============================================================

import subprocess, time, json, sys, os, traceback
import threading, queue, re

REPO = "/kaggle/working/olympiad-math-grpo"
results = {}
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # 消除显存碎片
T_GLOBAL0 = time.time()

def save_results():
    with open("/kaggle/working/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

def parse_output(lines):
    # Parse per-step metrics + per-eval-set accuracy from the training stdout
    curve, final, per_eval = [], {}, {}
    surprisal_curve, zerovar_curve, len_curve = [], [], []
    for line in lines:
        line = line.strip()
        if line.startswith("[step"):
            body = line.split("] ", 1)[-1] if "] " in line else ""
            vals = {}
            for part in body.split(" | "):
                k, _, v = part.partition("=")
                try:
                    vals[k] = float(v)
                except ValueError:
                    pass
            final.update(vals)   # accumulate: keep keys from every step
            if "mean" in vals:
                curve.append(vals["mean"])
            if "mean_sampled_token_surprisal" in vals:
                surprisal_curve.append(vals["mean_sampled_token_surprisal"])
            if "zero_variance_group_ratio" in vals:
                zerovar_curve.append(vals["zero_variance_group_ratio"])
            if "mean_completion_tokens" in vals:
                len_curve.append(vals["mean_completion_tokens"])
        elif line.startswith("[eval:"):
            spec = line[len("[eval:"):].split("]", 1)[0]        # e.g. olympiad
            vals = {}
            for _k in ("accuracy", "format_rate"):
                _m = re.search(_k + r"=(-?[0-9.]+)", line)
                if _m:
                    try:
                        vals[_k] = float(_m.group(1))
                    except ValueError:
                        pass
            _m = re.search(r"pass@(\d+)=(-?[0-9.]+)", line)
            if _m:
                vals[f"pass@{_m.group(1)}"] = float(_m.group(2))
            if vals:
                per_eval[spec] = vals
    final["_surprisal_curve"] = surprisal_curve
    final["_zerovar_curve"] = zerovar_curve
    final["_len_curve"] = len_curve
    final["_per_eval"] = per_eval
    return curve, final, per_eval

def record(algo, t0, lines, status, err_tail):
    curve, final, per_eval = parse_output(lines)
    oly = per_eval.get("olympiad", {})
    ind = per_eval.get("math_l45", {})
    results[algo] = {
        "minutes": round((time.time() - t0) / 60, 1),
        "status": status,
        "olympiad_accuracy": oly.get("accuracy"),      # 分布外（头条）
        "olympiad_format_rate": oly.get("format_rate"),
        "math_l45_accuracy": ind.get("accuracy"),     # 分布内
        "math_l45_format_rate": ind.get("format_rate"),
        "final_reward": final.get("mean"),
        "surprisal": final.get("mean_sampled_token_surprisal"),
        "surprisal_curve": final.get("_surprisal_curve", []),
        "zero_var_curve": final.get("_zerovar_curve", []),
        "len_curve": final.get("_len_curve", []),
        "clip_fraction": final.get("clip_fraction"),
        "zero_var_ratio": final.get("zero_variance_group_ratio"),
        "mean_completion_tokens": final.get("mean_completion_tokens"),
        "reward_curve": curve,
        "stderr_tail": err_tail,
    }
    save_results()
    print(f"\n[summary] {algo}: status={status} olympiad_acc={results[algo]['olympiad_accuracy']} "
          f"math_l45_acc={results[algo]['math_l45_accuracy']} steps={len(curve)} "
          f"({results[algo]['minutes']} min)", flush=True)

for algo in PRESETS:
    if time.time() - T_GLOBAL0 > GLOBAL_DEADLINE_H * 3600:
        print(f">>> [skip] {algo}: 已过全局截止，把剩余额度留给出图", flush=True)
        results[algo] = {"status": "skipped_global_deadline", "reward_curve": []}
        save_results()
        continue

    print(f"\n{'='*70}\n>>> preset: {algo}\n{'='*70}", flush=True)
    cmd = [sys.executable, "scripts/train.py",
           "--algorithm", algo,
           "--base-model", BASE_MODEL,
           "--dataset", DATASET,
           "--eval-dataset", EVAL_DATASET,
           "--train-samples", str(TRAIN_SAMPLES),
           "--eval-samples", str(EVAL_SAMPLES),
           "--max-steps", str(MAX_STEPS),
           "--group-size", str(GROUP_SIZE),
           "--groups-per-step", str(GROUP_PER_STEP),
           "--max-new-tokens", str(MAX_NEW_TOKENS),
           "--ppo-epochs", str(PPO_EPOCHS),
           "--learning-rate", LEARNING_RATE,
           "--overlong-cache", str(OVERLONG_CACHE),
           "--device", "cuda",
           "--use-swanlab", "false",
           "--run-name", algo]
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, cwd=REPO,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL,
                                text=True, bufsize=1, env=env)
        out_q, err_lines = queue.Queue(), []

        def pump_out():
            for line in proc.stdout:
                out_q.put(line.rstrip("\n"))
            out_q.put(None)

        def pump_err():
            for line in proc.stderr:
                err_lines.append(line.rstrip("\n"))

        threading.Thread(target=pump_out, daemon=True).start()
        threading.Thread(target=pump_err, daemon=True).start()

        stdout_lines, last_active, kill_reason = [], time.time(), None
        while True:
            try:
                line = out_q.get(timeout=15)
            except queue.Empty:
                line = "__timeout__"
            now = time.time()
            if line is None:
                break
            if line != "__timeout__":
                print(line, flush=True)
                stdout_lines.append(line)
                last_active = now
                continue
            if now - last_active > QUIET_KILL_MIN * 60:
                kill_reason = f"watchdog: {QUIET_KILL_MIN} 分钟无输出，判定挂死"
                break
            if now - t0 > BUDGET_MIN * 60:
                kill_reason = f"budget: 超过 {BUDGET_MIN} 分钟（保留已训练曲线）"
                break

        if kill_reason:
            print(f"\n=== [KILL] {algo}: {kill_reason} ===", flush=True)
            proc.kill()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        time.sleep(1.0)

        if proc.returncode != 0 or kill_reason:
            print(f"=== [STDERR tail] {algo} 最后 40 行 ===", flush=True)
            print("\n".join(err_lines[-40:]) or "(stderr 为空)", flush=True)
            status = (f"partial: {kill_reason}" if kill_reason and kill_reason.startswith("budget")
                      else f"dead: {kill_reason or f'exit code {proc.returncode}'}")
            record(algo, t0, stdout_lines, status, "\n".join(err_lines[-20:]))
            continue

        if err_lines:
            print(f"=== [STDERR tail] {algo} 最后 5 行（正常完成，仅警告） ===", flush=True)
            print("\n".join(err_lines[-5:]), flush=True)
        record(algo, t0, stdout_lines, "ok", "\n".join(err_lines[-5:]))

    except Exception as e:
        print(f"\n=== [FATAL] preset: {algo} ===", flush=True)
        traceback.print_exc()
        results[algo] = {"status": f"fatal: {e}", "reward_curve": [],
                         "minutes": round((time.time() - t0) / 60, 1)}
        save_results()
        continue

print(f"\n{'='*70}")
print(f"全部结束，总耗时 {(time.time() - T_GLOBAL0) / 3600:.2f} h")
print("结果 -> /kaggle/working/ablation_results.json")
print(f"{'='*70}", flush=True)
""")

code(r"""# Cell 7: 汇总表 + 曲线图（reward / surprisal / 零方差组占比 / 补全长度）
import json, os
import matplotlib.pyplot as plt

with open("/kaggle/working/ablation_results.json") as f:
    results = json.load(f)

ORDER = ["grpo", "clip_higher", "dynamic", "token_level", "overlong", "dapo", "entropy_reg"]
ORDER = [a for a in ORDER if a in results] + [a for a in results if a not in ORDER]

print("=" * 96)
print(f"{'algorithm':14s} {'oly_acc':>8s} {'math_acc':>9s} {'fmt':>7s} "
      f"{'zerovar':>8s} {'len':>6s} {'surprisal':>10s} {'min':>6s}  status")
print("-" * 96)
for a in ORDER:
    r = results[a]
    def f_(v, p=4):
        return f"{v:.{p}f}" if isinstance(v, (int, float)) else "  -  "
    print(f"{a:14s} {f_(r.get('olympiad_accuracy')):>8s} {f_(r.get('math_l45_accuracy')):>9s} "
          f"{f_(r.get('olympiad_format_rate')):>7s} {f_(r.get('zero_var_ratio')):>8s} "
          f"{f_(r.get('mean_completion_tokens'),1):>6s} {f_(r.get('surprisal')):>10s} "
          f"{str(r.get('minutes','-')):>6s}  {r.get('status','-')}")
print("=" * 96)

ref = next((a for a in ORDER if results[a].get("reward_curve")), None)
if ref:
    n = len(results[ref]["reward_curve"])
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    panels = [("reward_curve", "reward (shaped)", axes[0][0]),
              ("surprisal_curve", "surprisal (探索度)", axes[0][1]),
              ("zero_var_curve", "零方差组占比 (难题特有)", axes[1][0]),
              ("len_curve", "平均补全长度", axes[1][1])]
    for key, title, ax in panels:
        for a in ORDER:
            c = results[a].get(key) or []
            if c:
                ax.plot(range(len(c)), c, marker="o", ms=2.5, label=a)
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Olympiad-Math GRPO/DAPO — seven-arm ablation (Qwen2.5-1.5B, T4)")
    fig.tight_layout()
    fig.savefig("/kaggle/working/ablation_summary.png", dpi=110)
    print("\n曲线图 -> /kaggle/working/ablation_summary.png")
print("\n把 ablation_results.json 和 ablation_summary.png 下载下来，填进 README 的结果表。")
""")


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "kaggle_train.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()

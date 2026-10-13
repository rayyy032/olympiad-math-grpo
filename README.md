# Olympiad-Math GRPO/DAPO · 竞赛级数学推理的失效机制分析

用纯 PyTorch + Transformers + PEFT 从零手写 GRPO 训练循环，在 **Qwen2.5-1.5B-Instruct** 上做数学推理 RL，训练集为 **MATH level 4-5**（竞赛级，3994 题），并用 **AIME / AMC / MATH-L5**（894 题，Kaggle AIMO 竞赛同源题目）做**分布外压力测试**。

核心问题不是"GRPO 能不能提升分数"，而是：**当题目从小学难度跃升到竞赛难度时，GRPO / DAPO 的哪些机制从"可选项"变成"必需品"，以及 RL 究竟学到了推理能力还是题目分布。**

不依赖 TRL / verl / OpenRLHF —— 组内相对优势、PPO-clip 目标、KL、LoRA 注入、left-padding 批量生成、response-span log-prob 切片，全部手写，每个张量的形状都可断点检查。

## 为什么换成竞赛级难题：GSM8K vs MATH-L45 的三个质变

同一套 GRPO 代码，在 GSM8K（小学应用题）上 DAPO 的四项改进是**锦上添花**（基线已达 32%）；在 MATH L4-5 / AIME 这类竞赛题上，它们变成**雪中送炭**，因为难题天然引发三个质变：

| 现象 | GSM8K（小学题） | 竞赛级难题 | 受影响的 DAPO 组件 |
|---|---|---|---|
| **零方差组占比** | ≈ 0%（8 条采样总有对有错） | 显著升高：8 条采样可能**全错** → advantage 恒为 0 → 梯度消失 | Dynamic Sampling（从可选变必需） |
| **推理链长度** | 短，很少触顶 | 长，频繁撞 `max_new_tokens` 截断 → reward 悬崖 | Overlong Reward Shaping |
| **优势分布** | 集中 | 极端稀疏（少数样本才对） | Clip-Higher（放宽优势侧上界） |

**这正是本项目的研究切入点**：把 GRPO 放在它真正会失效的压力区间里，逐项验证每个组件的实际作用。

## 实验设计：训练/评测分离 + 分布外泛化

```
训练集  math_l45   = MATH level 4-5               3994 题（竞赛级）
        ↓  GRPO / DAPO 七组单变量消融
评测集  math_l45   = MATH level 4-5 test split    2538 题（分布内，衡量提升）
        olympiad   = AIME(90) + AMC(83) + MATH-L5(721)  894 题（分布外，衡量泛化）
```

- **分布内**：训练后同分布难度是否真的提升。
- **分布外**：`AIME / AMC` 是更难的竞赛题（Kaggle AIMO 同源），量化"RL 学到的是推理能力，还是 MATH 的数据分布"——这是回答 reward hacking 与泛化质疑的关键证据。
- **难度分层**：评测按 `per_source`（aime / amc / math_l5 / math level）自动拆解，定位能力边界。

## DAPO 四项：公式与实现

### 1. Clip-Higher（非对称裁剪）

$$
r_i(\theta)=\frac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{old}}(o_i\mid q)},\qquad
\hat A_i = \frac{r_i^{reward}-\mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r)+\varepsilon}
$$

$$
\mathcal L_{policy}=-\mathbb E\Big[\min\big(r_i\hat A_i,\ \mathrm{clip}(r_i,\,1-\varepsilon_{low},\,1+\varepsilon_{high})\hat A_i\big)\Big]
$$

低概率 token（探索的来源）一旦被正 advantage 放大，很容易撞到对称上界 `1+ε=1.2` 被裁掉。DAPO 把上界放宽到 `1+eps_high=1.28`，下界保持 `1-eps_low=0.8`，只放宽"变强"的方向。**难题上正优势样本极稀疏，这个放宽更关键。** 实现见 `src/losses.py: grpo_loss(eps_low, eps_high)`。

### 2. Dynamic Sampling（动态采样补齐）

组内 reward 全相同时 `std=0`、advantage 全 0，这批样本不产生有效梯度。**难题下这一现象从偶发变为高发**，开启后每个 batch 丢弃零方差组并继续补采，直到凑满 `groups_per_step` 或达到 `dynamic_max_attempts` 倍上限。训练日志中的 `zero_variance_ratio` / `effective_group_ratio` / `oversample_ratio` 即此机制的诊断量。实现见 `scripts/train.py: sample_groups`。

### 3. Token-level Loss（逐 token 损失）

GRPO 原始实现按样本均值再对 batch 均值，相当于给长回答隐式加权。竞赛题的推理链长度方差极大，这个偏置被放大。改为全 batch 有效 response token 统一求均值：

$$
\mathcal L=\frac{1}{\sum_i|o_i|}\sum_i\sum_{t}\big[\min(\cdot)\big]_{i,t}
$$

实现见 `src/losses.py: grpo_loss(token_level=True)`。

### 4. Overlong Reward Shaping（超长软惩罚）

对达到长度预算被截断的回答，硬奖励（直接给 0/-1）会让 reward 出现悬崖。**竞赛题推理链长，截断率远高于 GSM8K**，软惩罚因此从微调项变为关键项。改为分段线性（`src/rewards.py: soft_overlong_penalty`）：

$$
penalty(len)=
\begin{cases}
0, & len < L_{max}-L_{cache}\\[4pt]
-\dfrac{len-(L_{max}-L_{cache})}{L_{cache}}, & L_{max}-L_{cache}\le len < L_{max}\\[8pt]
-1, & len \ge L_{max}
\end{cases}
$$

进入软区后惩罚随长度线性加深，到预算上限饱和为 -1，不再有突变。

## 扩展 A：熵正则（探索保持）

熵坍缩只靠 Clip-Higher 缓解不够彻底，额外加显式熵奖励项，用采样 token 的平均 surprisal 近似策略熵：

$$
\mathcal L_{ent}=-\mathrm{ent\_coef}\cdot\frac{1}{|o|}\sum_t\big(-\log\pi_\theta(o_t\mid q,o_{<t})\big)
$$

对应第七组消融 `entropy_reg`（在 DAPO 全开基础上叠加），实现见 `src/losses.py: compute_surprisal`。**难题上策略更容易过早收敛到"放弃式"短输出，熵正则的对照价值更高。**

## 扩展 B：Reward-Hacking 监控

规则奖励（答案匹配 + 格式分）天然可被 hack。三个检测器**只记录指标、不进 reward**，用于在曲线异常前定位投机行为（`src/rewards.py: detect_reward_hacking`）：

- `length_exploit`：长回答得分显著高于短回答（啰嗦换分）
- `format_only_hack`：格式分占比异常升高（会写 `\boxed{}` 但答案错误）
- `equivalence_bypass`：等价判定通过但字符完全不符（等价逻辑漏洞）

竞赛题上"写不出答案、只凑格式分"是最典型的 hack 路径，这三个指标是防守手段。

## Reward 设计：math-verify 数值等价

答案判定三级（`src/rewards.py: answers_equivalent`）：

1. 精确字符串匹配
2. 纯数值 fast-path：剥离 `$`/`,`/空白 → `float` 解析 → `math.isclose`（覆盖 `1e3 == 1000`、`42.0 == 42`，并兼容 AMC 数据集的 float 型答案）
3. math-verify 符号等价（`\frac{1}{2} == 0.5`）

`\boxed{}` 提取支持任意深度嵌套括号并取**最后一个**完整匹配（`extract_last_boxed`）。最终 reward = 答案对错（0/1）+ 格式分（`format_weight`）+ overlong 软惩罚。

## 数据管线

| 数据集 | 来源 | 规模 | 用途 |
|---|---|---|---|
| `math_l45` | `EleutherAI/hendrycks_math`（level 4-5） | 训练 3994 / 测试 2538 | 训练 + 分布内评测 |
| `aime` | `AI-MO/aimo-validation-aime` | 90 | 分布外压力测试 |
| `amc` | `AI-MO/aimo-validation-amc` | 83 | 分布外压力测试 |
| `math_l5` | `AI-MO/aimo-validation-math-level-5` | 721 | 分布外压力测试 |
| `olympiad` | 上述三者合并 | 894 | 分布外总评 |

- MATH 的 ground truth 藏在 solution 的 `\boxed{}` 里，数据层用 `extract_last_boxed` 抽取（直接传整段 solution 会导致答案永远匹配不上——原仓库的隐性 bug）。
- AMC 的 `answer` 是 float（如 `142.0`），数据层归一化为 `"142"`，由数值等价判定兜底。
- 全部支持本地 JSONL 兜底：`--dataset /path/to/any.jsonl`（`{"question","answer"}`）。

## 工程细节：left padding 与 response span

decoder-only 模型批量生成必须 **left padding**（右侧 padding 会导致生成位置错乱）。本项目的批量流水线（`src/generation.py`）：

- 生成时 `padding_side="left"`，pad token 用 `<|endoftext|>`（与 eos 区分）
- 每行记录 response 的 `(start, end)` span（扫描 pad + eos 定位真实回答边界）
- log-prob 重算按 span 切片 gather，剔除 pad 位置的 logits 污染
- 训练全程零 "right-padding may cause unexpected behavior" 警告

配置上 `apply_preset` 自动钳制 `overlong_cache < max_new_tokens`，小预算冒烟不会触发非法参数。

## 七组单变量消融

| algorithm | clip_higher | dynamic_sampling | token_level | overlong_shaping | entropy_reg |
|---|:-:|:-:|:-:|:-:|:-:|
| `grpo`（基线） | | | | | |
| `clip_higher` | ✅ | | | | |
| `dynamic` | | ✅ | | | |
| `token_level` | | | ✅ | | |
| `overlong` | | | | ✅ | |
| `dapo`（四项全开） | ✅ | ✅ | ✅ | ✅ | |
| `entropy_reg`（DAPO+熵） | ✅ | ✅ | ✅ | ✅ | ✅ |

七组除算法开关外**全部超参一致**（preset 表见 `src/config.py`），保证差异只来自算法本身。同一套开关表在 GSM8K 与 MATH-L45 上复用，构成了难度维度的对照。

## 实测结果

> 结果由 `scripts/run_ablations.sh` 在单张 T4 上跑出，每个 run 落盘 `outputs/ckpt_<algo>/eval_result.json`。运行后按 algorithm × eval-set 汇总填入下表。

**分布外总评（olympiad: AIME+AMC+MATH-L5）**

| algorithm | olympiad acc | aime | amc | math_l5 | format_rate | 零方差组占比 | 平均补全长度 | 耗时 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `base`（0-shot 未训练） | | | | | | — | — | — |
| `grpo` | | | | | | | | |
| `+ clip_higher` | | | | | | | | |
| `+ dynamic_sampling` | | | | | | | | |
| `+ token_level` | | | | | | | | |
| `+ overlong_shaping` | | | | | | | | |
| `dapo`（四项全开） | | | | | | | | |
| `entropy_reg`（DAPO+熵） | | | | | | | | |

**分布内（math_l45 test）**

| algorithm | math_l45 acc | format_rate | 平均补全长度 |
|---|:-:|:-:|:-:|
| `base`（0-shot） | | | |
| `grpo` | | | |
| `dapo` | | | |
| `entropy_reg` | | | |

**对照基线（原 GSM8K 版本，Qwen2.5-0.5B，100 步）**

用于对照"难度跃升前后 DAPO 组件重要性的变化"：vanilla GRPO 32% → 单项最优 41%；DAPO 全开在 0.5B 上复现熵坍缩（零方差组 100%、长度归零），加熵正则后恢复 35%。原始数据见原仓库 `results/ablation_results.json`。

> 注：本批为小规模受控消融（受单卡预算限制），数字用于**组间相对比较与机制定位**，而非绝对 SOTA 声明。

## 快速开始

```bash
# 安装
pip install -r requirements.txt

# 27 项边界单测（boxed 提取 / 数值等价 / overlong 边界 / 端到端打分）
python -m pytest tests/ -q

# CPU 冒烟（合成数据，0.5B + LoRA，验证整条训练流水线）
python scripts/train.py --algorithm dapo --dataset synthetic \
    --train-samples 20 --max-steps 2 --device cpu

# 正式训练：竞赛级难题，分布外评测
python scripts/train.py --algorithm entropy_reg \
    --dataset math_l45 --eval-dataset olympiad,math_l45 \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --train-samples 3000 --eval-samples 150 \
    --max-new-tokens 384 --max-steps 150

# 七组消融一键跑（BASE_MODEL / TRAIN_SET / EVAL_SET 等可用环境变量覆盖）
bash scripts/run_ablations.sh

# 没有 GPU？Kaggle 免费 T4 一键跑（上传 kaggle_train.ipynb → Run All）
```

训练曲线与诊断面板（reward / accuracy / surprisal / clip_fraction / 零方差组占比 / hack rates）默认落盘到 `outputs/` 与 `swanlog/`（SwanLab 本地模式，无需注册账号）。

## 项目结构

```
src/
  losses.py       # GRPO 目标：非对称 clip / token-level / KL / 熵正则 / 诊断统计
  rewards.py      # boxed 提取、三级数值等价、overlong 软惩罚、hack 检测
  generation.py   # left-padding 批量生成 + response-span log-prob 切片
  config.py       # dataclass 配置 + 七组 preset + CLI 覆盖
  data.py         # MATH-L45 / AIME / AMC / MATH-L5 / GSM8K 数据构建 + 离线合成集
  evaluation.py   # 贪心解码评测（accuracy / format / per_source 分层）
  diagnostics.py  # 训练指标聚合
  agents/         # Qwen + LoRA 策略封装、frozen reference
scripts/
  train.py        # 统一训练入口（dynamic sampling 主循环 + 多评测集）
  run_ablations.sh
tests/
  test_reward.py  # 27 项边界单测
```

## 诚实边界

- 本项目的贡献是**在竞赛级难题上做受控消融与失效机制定位**，不是刷到 SOTA。小模型 + 小步数受单卡预算约束。
- 数据来自公开学术数据集（Hendrycks MATH、AI-MO 官方验证集），未做任何标注造假。
- 与 Kaggle AIMO 竞赛的关系：使用其**同源**题目集（AI-MO 验证集）做压力测试，不是竞赛参赛成绩，README 与简历均不暗示达到获奖方案水平。

## Acknowledgments

- GRPO 训练循环骨架参考 [Feld-maxiu/bachelor-thesis-rl](https://github.com/Feld-maxiu/bachelor-thesis-rl)；算法扩展、消融设计与失效分析为本仓库工作。
- DAPO 算法来自 [BytedTsinghua-SIA/DAPO](https://github.com/bytedance/DAPO) 论文 *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*。
- 评测题目来自 [AI-MO](https://huggingface.co/AI-MO) 官方验证集与 [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)。

## License

Apache-2.0

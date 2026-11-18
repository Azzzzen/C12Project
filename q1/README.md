# Q1：从两体阈值到三体格点模型

本目录只处理题目 Q1。Q2 的 measurements.txt 不参与计算，Q3 的恒星反应率也不在本报告范围内。

## 文件

- `model.py`：周期最近邻动能、最小镜像势能、交换与正方格点对称操作。
- `run.py`：完整相对坐标空间中的低能本征态、参数扫描、有括区间求根；每次求解后保存 JSON。
- `validate.py`：自由谱、厄米性、对称算子对易、小格子稠密对角化、Hellmann–Feynman 与容差检查。
- `illustrate_matrices.py`：展开两体矩阵及三体分块矩阵，生成跳跃与稀疏结构图，并独立核对张量积和分块表示。
- `analyze.py`：汇总 CSV、生成图和报告数值表，并审计全部数值结果。
- `data/`：原始计算结果、检查结果及基准波函数。
- `figures/`：Python 生成的 PDF、SVG 和 PNG 图。
- `report/q1_report_zh.tex`、`report/q1_report_en.tex`：中文 / 英文 LaTeX 报告（内容相同），及对应 PDF。

## 可复现运行

需要 Python 3、NumPy、SciPy、Matplotlib。使用单线程 BLAS 避免线程开销。

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r q1/requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/carbon-mpl
python q1/run.py --L 8 12 16 20 --cs 3.5 5 7 --roots
python q1/run.py --L 8 12 16 20 --roots
python q1/run.py --L 24 28 32 --cs 3.5 4 4.25 4.5 5 7 --roots
python q1/run.py --L 36 40 --cs 4.28 4.31 5 --roots --k 2
python q1/validate.py
python q1/analyze.py
python q1/illustrate_matrices.py
cd q1/report
for f in q1_report_zh q1_report_en; do
  xelatex -interaction=nonstopmode -halt-on-error $f.tex
  xelatex -interaction=nonstopmode -halt-on-error $f.tex
done
```

已有 JSON 按 `(L,c)` 缓存；更改模型或求解精度时，应先将旧的 `data/L*.json` 移到备份目录，避免混用不同模型结果。较大格子占用更多内存和时间。

## 约定与术语

| 符号 | 唯一定义 |
|---|---|
| `L` | 每个方向的格点数；物理边长为 `La` |
| `t` | ℏ²/(2ma²)，数值计算取 1 |
| `E2 = E_thr` | 两体基态能量；题目定义的两体加静止粒子阈值 |
| `E31, E32` | 玻色交换对称且属于正方格点 A1 表示的最低两个态 |
| `ER` | `E32 - E2`，不是三体激发能，也不是实核共振能 |
| `B` | `E2 - E31`，不是总结合能 |
| `c_star(L)` | 有限格子上 `ER = 0` 的耦合强度 |

## 图表设计与证据边界

采用 Python 定量多面板图。能谱与相对阈值曲线展示作用强度依赖；穿越点与固定参数能量展示有限体积效应；两态边缘概率展示空间结构。结论以实际计算为准，不预设无限体积穿越点。

输出为约 180 mm 宽的矢量 PDF/SVG 和 PNG 预览，文字可编辑。所有曲线来自确定性本征值计算；连线仅辅助阅读，不代表额外采样。不存在抽样置信区间，数值求根容差不能当作有限体积误差。概率图使用统一色标、完整周期盒子和明确的对数变换，不做局部图像增强。

图中的有限格子穿越与无限体积物理解离阈值必须区分。Q1 不需要贝叶斯推断；项目后续 Q2 才要求推断未知的相互作用强度。

## 报告结构

`report/q1_report_zh.pdf` 与 `report/q1_report_en.pdf` 内容一致（各约 6 页），按“先原理、后数值”组织，便于讲解：

1. 题目原文与结论速览。
2. 理论推导：模型 → 格点动能 → 平移不变性与相对坐标（两体 `2t`、阈值、三体斜向跳跃）→ 对称性与 `0+` 定义 → Hellmann–Feynman → 自由谱与有限体积两个解析判据。
3. 数值方法：Lanczos 与残差界、选态算法、Brent 求根、`L` 收敛。
4. 验证表。
5. 结果：基准、`c` 依赖、有限体积与 `c*`、空间结构。
6. 结论与适用范围。

报告中的数值来自脚本自动生成的 `numbers.tex`、`convergence_table.tex`，不要手动修改。`two_body_matrix.tex`、`three_body_blocks.tex` 为 `L=3` 矩阵展开，当前报告未引用。

# MedRefocus：MSD Pancreas 实现

本仓库公开 MedRefocus 在 MSD 胰腺 CT 上的实现，包括候选区域生成、37 维 UCRP 排序、四通道 3D Local Expert、118 维 PCSC 提交与原始空间评价。

每例只调用一次局部专家。Auxiliary 参与选区；专家直接输入为 CT 与 Global 的概率和硬标签。局部提案只新增肿瘤体素，PCSC 接受完整提案或保留完整原始 Global 预测。新增体素仍可能是假阳性。

## 使用

建议 Python 3.10/3.11，先安装适合本机的 PyTorch，再执行：

```bash
git clone https://github.com/LIN021/MedRefocus-MSD.git
cd MedRefocus-MSD
python -m pip install -e '.[dev]'
medrefocus --help
python -m pytest
```

下载 MSD Task07_Pancreas 后，按以下顺序操作：

1. `prepare-msd` 转换数据，`split` 生成外层五折和内部选模划分。
2. 为每个外层折分别训练 Global 与 Auxiliary；只使用该折外层训练患者。将预测和概率恢复到原始 CT 网格。
3. `manifest` 生成患者输入清单，`check` 检查几何与概率。
4. `train` 在内部 fit/selection 划分上拟合并选择模型；外层测试标签不进入训练。
5. `predict --role test` 进行无参考标签的推理。
6. `evaluate` 对冻结的最终结果评价。

完整命令见 [英文训练指南](docs/TRAINING.md)。[数据格式](docs/DATA.md)、[推理说明](docs/INFERENCE.md)与[评价定义](docs/EVALUATION.md)分别解释输入、输出和统计规则。

## 论文结果与代码范围

论文 MSD 汇总：Dice 51.21% → 55.47%，NSD@2 mm 49.98% → 53.72%，完整漏检 47 → 19；恢复 28 例，其中 12 例最终 Dice ≥25%，3 例 ≥50%。这些是论文已报告值，仓库不包含原始 CT、患者级结果或训练权重。重新训练将生成新的模型和实测结果。默认生成的划分用于新的确定性实验；有原始划分时可通过 `--outer-splits` 导入。

代码采用 MIT 许可证。仓库的自动测试仅使用小型合成数据，在 CPU 上检查流程、特征维度、add-only 和拒绝回退等行为。

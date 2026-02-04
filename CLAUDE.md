# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目规则

## 项目概述

这是论文 "The Power of Architecture: Deep Dive into Transformer Architectures for Long-Term Time Series Forecasting" (arXiv:2507.13043) 的官方代码库。项目系统性研究了 Transformer 架构在长期时间序列预测(LTSF)任务中的应用。

**核心发现**:

- 双向注意力与联合注意力机制表现最佳
- 完整的预测聚合策略(跨越回看窗口和预测窗口)能提高准确性
- 直接映射范式优于自回归建模
- BatchNorm 对异常值较多的时间序列效果更好,LayerNorm 对更平稳的时间序列表现更优

## 快速开始

### 运行完整实验

```bash
# 在所有8个数据集上运行所有模型
bash scripts/all_models/etth1.sh
bash scripts/all_models/etth2.sh
bash scripts/all_models/ettm1.sh
bash scripts/all_models/ettm2.sh
bash scripts/all_models/electricity.sh
bash scripts/all_models/weather.sh
bash scripts/all_models/traffic.sh
bash scripts/all_models/illness.sh
```

### 运行单个模型实验

```bash
# 使用预配置脚本
bash scripts/Transformer/etth1_LN.sh

# 或直接运行 Python 脚本
python run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model Transformer \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 3 \
  --norm layer \
  --run_train --run_test
```

## ⚠️ 关键约束和注意事项

### 1. 脚本执行方式

**必须使用 `bash` 而不是 `sh` 运行脚本**:

```bash
bash script.sh  # ✓ 正确
sh script.sh    # ✗ 会报错
```

### 2. Patch 配置规则

- `patch_len` 和 `stride` **必须相同**(非重叠 patch)
- `seq_len` 和 `pred_len` **必须**是 `patch_len` 的倍数
- 默认配置: `patch_len=16`, `stride=16`

### 3. 层数配置(保证参数量一致)

不同架构类型的层数配置必须遵循以下规则以保证公平对比:

| 架构类型        | e_layers | d_layers | 总层数 |
| --------------- | -------- | -------- | ------ |
| Encoder-only    | 6        | 0        | 6      |
| Decoder-only    | 0        | 6        | 6      |
| Encoder-Decoder | 3        | 3        | 6      |
| Double架构      | 6        | 6        | 12     |

### 4. 超参数一致性

所有模型应使用相同的超参数以确保公平对比:

- `d_model=512`
- `n_heads=8`
- `d_ff=2048`
- `patch_len=16`, `stride=16`

## 代码架构

### 核心模块结构

```
data_provider/          # 数据加载和预处理
├── data_factory.py     # 数据集工厂
└── data_loader.py      # Dataset 类实现

exp/                    # 实验管理
├── exp_basic.py        # 实验基类
└── exp_main.py         # 训练/测试/评估主逻辑

layers/                 # 可复用网络层
├── Transformer_EncDec.py      # Encoder/Decoder 层
├── SelfAttention_Family.py    # 注意力机制
├── Embed.py                   # 位置/时间/Token 嵌入
├── PatchTST_backbone.py       # PatchTST 骨干网络
└── RevIN.py                   # 可逆实例归一化

models/                 # 模型实现(22个变体)
├── Transformer.py             # 标准 Transformer
├── Encoder.py                 # Encoder-only 架构
├── Encoder_overall.py         # 整体聚合 Encoder
├── Encoder_zeros_*.py         # 零填充变体
├── Masked_encoder_*.py        # 掩码变体
├── Decoder.py                 # Decoder-only 架构
├── Decoder_autoregressive.py  # 自回归 Decoder
├── Prefix_decoder.py          # 前缀 Decoder
├── Double_encoder.py          # 双 Encoder
├── Double_decoder.py          # 双 Decoder
└── [其他基线模型]

scripts/                # 实验脚本(119个)
├── all_models/         # 所有模型完整对比实验
└── [各模型专用目录]

utils/                 # 工具函数
├── tools.py           # 学习率调整、早停等
├── metrics.py         # 评估指标(MSE/MAE)
└── timefeatures.py    # 时间特征编码
```

### 数据流

```
CSV 数据 → data_loader (Dataset 类)
    ↓
数据预处理(归一化、特征工程)
    ↓
DataLoader (批量化)
    ↓
Model.forward()
    ↓
RevIN 归一化 → Patch 切割 → Embedding
    ↓
Encoder/Decoder 处理
    ↓
预测头 → RevIN 反归一化
    ↓
损失计算与反向传播
```

## 主要模型架构类型

### Encoder-only 系列

这6个模型主要区别在于如何处理预测窗口的占位符以及如何将 patch 表示映射回原始序列长度。

#### 基础模型

- **`Encoder`**: 基础 Encoder + 展平预测头
- **`Encoder_overall`**: 整体聚合(回看+预测窗口),同时对两个窗口进行建模

#### 填充策略变体

这些模型在预测窗口使用不同的占位符填充方式:

**1. 零填充 (`Encoder_zeros_*.py`)**

```python
# 使用固定的零向量作为占位符
with torch.no_grad():
    self.mask = torch.zeros(1, d_model)  # 不可学习
```

- **不可学习**: mask token 始终为零,不参与梯度更新
- **简单直接**: 作为"空白"位置,让模型学习到"这里需要预测"
- **计算效率高**: 不需要额外的参数和梯度计算

**2. 掩码填充 (`Masked_encoder_*.py`)**

```python
# 使用可学习的 embedding 作为占位符
self.mask = nn.Embedding(1, d_model)  # 可学习参数
```

- **可学习**: mask token 作为可训练参数,通过反向传播自动优化
- **自适应**: 模型可以学习到最适合的占位符表示
- **更灵活**: 可以编码关于"需要预测"位置的先验知识

#### 映射策略变体

**1. 展平映射 (`*_flatten.py`)**

```python
# 先展平所有 patch,然后通过一个大的线性层映射
self.flatten = nn.Flatten(start_dim=-2)
self.projection = nn.Linear(output_patch_num * d_model, pred_len)

# 使用
output = self.flatten(enc_out)  # [bs*ch, output_patch_num, d_model] → [bs*ch, output_patch_num*d_model]
output = self.projection(output)  # [bs*ch, output_patch_num*d_model] → [bs*ch, pred_len]
```

- **全局映射**: 所有 patch 的表示通过一个大的线性层一次性映射到预测长度
- **参数共享**: 跨越所有 patch 的权重共享
- **计算效率**: 一次矩阵操作完成映射

**2. 逐 Patch 映射 (`*_no_flatten.py`)**

```python
# 先将每个 patch 单独映射,然后展平
self.projection = nn.Linear(d_model, patch_len)  # 每个 patch 独立映射
self.flatten = nn.Flatten(start_dim=-2)

# 使用
output = self.projection(enc_out)  # [bs*ch, output_patch_num, d_model] → [bs*ch, output_patch_num, patch_len]
assert output.shape[-1] * output.shape[-2] == pred_len
output = self.flatten(output)  # [bs*ch, output_patch_num, patch_len] → [bs*ch, pred_len]
```

- **局部映射**: 每个 patch 的 embedding 独立映射到 patch_len
- **权重共享**: 每个 patch 使用相同的线性层权重
- **保持局部性**: 保持了 patch 和输出之间的对应关系

#### 模型对比总结

| 模型                        | 填充方式     | 映射方式      | 特点                | 适用场景            |
| --------------------------- | ------------ | ------------- | ------------------- | ------------------- |
| `Encoder_zeros_flatten`     | 零填充       | 展平映射      | 最简单,全局映射     | 快速实验,基线对比   |
| `Encoder_zeros_no_flatten`  | 零填充       | 逐 patch 映射 | 保持局部性          | 需要保留 patch 结构 |
| `Masked_encoder_flatten`    | 可学习占位符 | 展平映射      | 自适应填充,全局映射 | 追求最佳性能        |
| `Masked_encoder_no_flatten` | 可学习占位符 | 逐 patch 映射 | 自适应填充,局部映射 | 平衡性能和结构化    |

**工作流程** (所有模型类似):

1. 输入序列进行 Patch 切割: `[batch, seq_len, channel]` → `[batch*channel, patch_num, patch_len]`
2. 对输入 patches 做 value embedding
3. 添加占位符(mask token)作为预测窗口的初始表示
4. 添加位置编码
5. 通过 Encoder 处理
6. 提取预测窗口部分的输出
7. 通过预测头映射到 `pred_len`

### Decoder-only 系列

- `Decoder`: 无交叉注意力的 Decoder
- `Decoder_autoregressive`: 自回归 Decoder
- `Prefix_decoder`: 前缀 Decoder

### Encoder-Decoder 系列

- `Transformer`: 标准 Transformer
- `Transformer_autoregressive`: 自回归版本
- `Transformer_no_patch`: 无 Patch 版本

### 双架构系列

- `Double_encoder`: 双 Encoder 架构
- `Double_decoder`: 双 Decoder 架构

### 基线模型

- `PatchTST`: Patch Time Series Transformer
- `Autoformer`: Auto-Correlation Transformer
- `Informer`: 稀疏注意力 Transformer
- `Linear`/`NLinear`/`DLinear`: 线性基线

## 关键参数说明

### 数据相关

- `--data`: 数据集名称(ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Weather, Traffic, Illness)
- `--root_path`: 数据根目录(默认 `./dataset/`)
- `--data_path`: 具体 CSV 文件名
- `--features`: 预测任务类型(M: 多变量对多变量, S: 单变量对单变量, MS: 多变量对单变量)
- `--seq_len`: 输入序列长度(必须是 patch_len 的倍数)
- `--pred_len`: 预测序列长度(必须是 patch_len 的倍数)

### 模型相关

- `--model`: 模型名称(必须是 models/ 中存在的类名)
- `--e_layers`: Encoder 层数
- `--d_layers`: Decoder 层数
- `--d_model`: 模型维度(默认 512)
- `--n_heads`: 注意力头数(默认 8)
- `--patch_len`: Patch 长度(默认 16)
- `--stride`: Patch 步长(实验中应设为 16)
- `--norm`: 归一化类型(batch 或 layer)
- `--revin`: 是否使用 RevIN(1: 启用, 0: 禁用)
- `--inverse`: 是否对预测结果进行逆归一化(用于计算真实尺度的指标)

### 训练相关

- `--batch_size`: 批大小(常用 32 或 128)
- `--train_epochs`: 训练轮数(实验中常用 20)
- `--learning_rate`: 学习率(默认 0.0001)
- `--gpu`: GPU 编号
- `--run_train`: 是否运行训练
- `--run_test`: 是否运行测试

## 数据集

项目支持8个基准数据集:

| 数据集      | 类型           | 特征              |
| ----------- | -------------- | ----------------- |
| ETTh1/ETTh2 | 电力变压器温度 | 小时级, 7个特征   |
| ETTm1/ETTm2 | 电力变压器温度 | 15分钟级, 7个特征 |
| Electricity | 电力负载       | 321个客户         |
| Weather     | 天气预测       | 21个气象特征      |
| Traffic     | 交通流量       | 862个传感器       |
| Illness     | 疾病相关       | 疾病预测数据      |

数据划分:

- 默认: 训练集 70%, 验证集 10%, 测试集 20%
- 可通过 `train_ratio` 参数调整

## 重要技术特性

### Patch 机制

- 非重叠 patch 切割(stride == patch_len)
- 输入变换: [batch, seq_len, channel] → [batch*channel, patch_num, patch_len]
- 每个 patch 独立处理

### 归一化策略

- **RevIN** (Reversible Instance Normalization): 默认启用,可学习参数
- **BatchNorm vs LayerNorm**: 通过 `--norm` 参数切换
- **subtract_last**: 减去均值或减去最后一个值

### 注意力机制

- `FullAttention`: 标准 O(L²) 复杂度注意力
- 多头注意力(n_heads=8 默认)
- 可配置 attention_dropout

### 预测范式

- **直接映射**: 一次性输出所有预测
- **自回归**: 递归生成预测(逐个时间步)

### 聚合策略

- **整体聚合**: Encoder_overall,跨越回看和预测窗口
- **Patch 级别聚合**: 在 patch 层面聚合信息
- **展平后线性映射**: 先展平再通过线性层

## 添加新模型

1. 在 `models/` 目录创建新文件:

```python
import torch
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, configs):
        super(YourModel, self).__init__()
        # 配置模型结构

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 前向传播
        # x_enc: 编码器输入 [batch, seq_len, channels]
        # x_mark_enc: 编码器时间标记 [batch, seq_len, mark_dim]
        # x_dec: 解码器输入 [batch, label_len+pred_len, channels]
        # x_mark_dec: 解码器时间标记 [batch, label_len+pred_len, mark_dim]
        return dec_out  # [batch, pred_len, channels]
```

2. 在 `exp/exp_main.py` 的 `_build_model()` 方法中注册:

```python
model_dict = {
    ...
    'YourModel': YourModel,
}
```

## 添加新数据集

1. 在 `data_provider/data_loader.py` 创建新 Dataset 类:

```python
from torch.utils.data import Dataset

class Dataset_YourData(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='data.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # 初始化参数

    def __read_data__(self):
        # 读取和预处理数据
        # 返回 self.df_x, self.df_y

    def __getitem__(self, index):
        # 返回单个样本
        return s_begin, s_end, r_begin, r_end, s_mask, r_mask

    def __len__(self):
        # 返回数据集大小
```

2. 在 `data_provider/data_factory.py` 注册:

```python
data_dict = {
    ...
    'your_data': Dataset_YourData,
}
```

## 实验输出

### 日志文件

```
./script_outputs/{model_id}_{seq_len}_{pred_len}/{model}_{norm}norm.log
```

### 模型检查点

```
./checkpoints/{setting}/checkpoint.pth
```

### 测试结果

- MSE 和 MAE 指标
- 可视化图表(如果启用)

## 依赖和环境

### 环境

该项目需要允许在指定 conda 环境下

```
conda activate tsinghua
```

### Python 依赖

```
numpy
matplotlib
pandas
scikit-learn
torch==1.11.0
```

### GPU 要求

- 支持 CUDA 的 GPU(推荐)
- 可通过 `--gpu` 参数指定 GPU 编号

## 开发建议

### 调试技巧

1. 使用小 batch_size 和少量 epoch 快速验证
2. 检查 tensor 形状是否正确(特别是 patch 操作)
3. 验证归一化是否正确应用
4. 确认层数配置符合架构类型要求

### 性能优化

1. 调整学习率(默认 0.0001)
2. 使用混合精度训练(如需要)
3. 调整 batch_size 以充分利用 GPU
4. 考虑梯度累积以处理更大的有效 batch size

### 常见问题排查

**问题 1**: 维度不匹配错误

- 检查 seq_len 和 pred_len 是否是 patch_len 的倍数
- 确认 stride 和 patch_len 是否相同

**问题 2**: 内存溢出

- 减小 batch_size
- 减小 seq_len 或 pred_len
- 减小模型维度(d_model)

**问题 3**: 训练不收敛

- 检查学习率是否过大
- 确认归一化是否正确应用
- 验证数据加载是否正确
- 检查层数配置是否合理

## 相关文件

- **主入口**: `run_longExp.py` (332行)
- **实验管理**: `exp/exp_main.py` (1312行)
- **数据加载**: `data_provider/data_loader.py`
- **核心层**: `layers/Transformer_EncDec.py`, `layers/SelfAttention_Family.py`
- **工具函数**: `utils/tools.py`, `utils/metrics.py`

## 论文引用

如果使用此代码库,请引用:

```bibtex
@article{shen2025power,
  title={The Power of Architecture: Deep Dive into Transformer Architectures for Long-Term Time Series Forecasting},
  author={Shen, Lefei and Chen, Mouxiang and Fu, Han and Ren, Xiaoxue and Wang, Xiaoyun Joy and Sun, Jianling and Li, Zhuo and Liu, Chenghao},
  journal={arXiv preprint arXiv:2507.13043},
  year={2025}
}
```

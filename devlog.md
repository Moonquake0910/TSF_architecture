# 开发日志

## Encoder_overall

### 命令行

```bash
cd TSF_architecture/
bash scripts/all_models/etth1.sh
bash scripts/all_models/electricity.sh
bash scripts/蒙东负荷/mongolia.sh
```

### 参数调整

### Feb 4
- 新增 mongolia.sh
    - seq_len (3x24) 72 三天
    - pred_len (7x24) 168 七天
    - patch 调成 12 匹配窗口大小
    - batch_size 32
    - train_ratio 0.6
    - features S 单一对单一
    - 用 --inverse 还原数据后再计算指标

### Jan 29 
- electricity.sh
    - 历史 256 预测 48 步
    - features MS 预测最后一列

#### Jan 28 
- etth1.sh
    - 增加打印 MAPE 指标
    - 历史 192 预测 48 步
- exp_main.py
    - 增加 inverse 参数，在可视化和计算指标前将数据还原到归一化之前的分布

#### Jan 22
- etth1.sh
    - 只使用 Encoder_overall 模型
    - 预测窗口只跑 96
    - --patience 3

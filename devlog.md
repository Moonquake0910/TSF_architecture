# 开发日志

## Encoder_overall

### 命令行

```bash
cd TSF_architecture/
bash scripts/all_models/etth1.sh
bash scripts/all_models/electricity.sh
```

### 参数调整

### Jasn 29 
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

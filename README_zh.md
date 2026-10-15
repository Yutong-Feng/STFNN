# STFNN：用于空气质量推断的时空场神经网络

[English](README.md)

论文 **"Spatio-Temporal Field Neural Networks for Air Quality Inference"**（IJCAI-24，AI for Good 专题）的官方 PyTorch 实现 [[论文]](https://www.ijcai.org/proceedings/2024/0803.pdf)。

空气质量推断是指在没有监测站的位置估计污染物浓度。STFNN 把空气质量看作定义在坐标 `(经度, 纬度, 时间)` 上的连续**时空场**，学习的是这个场的**梯度**，而不是场的取值。对每个目标位置，把邻居站点的观测值沿直线路径积分学到的梯度，"搬运"到目标处，再用学习到的权重融合各个邻居的估计。论文把这种结合全局时空场与局部时空图的框架称为 **金字塔推断（Pyramidal Inference）**：

$$\hat{y}^{tar} = \sum_{i=1}^{K} w_i \Big( y_i^{src} + \sum_{j=1}^{m} \mathbf{D}_{i,j} \cdot \vec{r}_i \Big)$$

| 论文模块 | 代码位置 |
|---|---|
| 时空编码：`[lng_z, lat_z]` + 日/周/月/年周期的 sin/cos | `layers.py:SimpleSTEmbedding` |
| 由邻居特征得到初始梯度 | `model.py:FeatFusion` |
| 环形估计（Ring Estimation）：在 `m` 个环形步上估计梯度差分 `D` | `model.py:RingEstimation`、`model.py:PyramidalInference.pyramidal_aggregation` |
| 邻居聚合（Neighbor Aggregation）：Transformer decoder + softmax 生成权重 `W` | `model.py:STAttention` |
| 消融：用静态的 IDW + SES 权重替代邻居聚合 | `model.py:PyramidalInference.idw_ses` |

## 安装

```bash
pip install torch numpy pandas tqdm
```

已在 Python 3.13、PyTorch 2.x 和 CUDA GPU 上测试；没有 GPU 时自动使用 CPU。

## 数据

全国空气质量数据集（2018-01-01 至 2018-12-31，逐小时，1085 个站点）存为 NumPy 数组 `air_2018.npy`，形状为 `(T, N, F) = (8760, 1085, 16)`：

| 索引 | 类别 | 特征 | 处理方式 |
|---|---|---|---|
| 0–5 | 空气质量 | PM2.5、PM10、NO2、CO、O3、SO2 | 按站点 z-score（目标列除外） |
| 6–10 | 气象（连续） | 降雨、温度、气压、湿度、风速 | 按站点 z-score |
| 11–12 | 气象（类别） | 风向、天气 | embedding |
| 13–15 | 时间上下文 | 小时、星期、是否节假日 | embedding |

站点坐标来自 `station_pm25_mean.csv`，只使用其中的 `longitude` 和 `latitude` 两列，行顺序必须与 `air_2018.npy` 的站点维一致。把两个文件放在同一目录下，并在 `configs/default.json` 里设置 `data_folder`。

`dataset.py:get_air_loader_normalizer` 中的预处理：
- 删除目标变量缺失站点超过 `drop_limit` 比例的时间步。
- 删除坐标重复的站点。
- 按时间顺序划分为 60% 训练、20% 验证、20% 测试。
- 每个样本随机遮蔽 `mask_ratio` 比例的站点，作为推断目标。
- 对每个目标，在间隔 `skip_step` 小时的 `window + 1` 个时间步上，各取 `k_top` 个最近的站点作为邻居。

## 运行

使用默认配置训练并测试：

```bash
python train.py
```

任何配置项都可以在命令行覆盖：

```bash
python train.py --label no2 --mask_ratio 0.5
```

日志和最佳模型（验证集 MAE 最低）保存在 `logs/<时间戳>/training.log` 与 `logs/<时间戳>/model_parameter.pth`。训练结束后，在测试集上以原始单位报告 MAE、MAPE、RMSE。

### 配置项

优先级：命令行 > `configs/model.json` > `configs/default.json`。

| 配置项 | 默认值 | 含义 |
|---|---|---|
| `label` | `pm25` | 推断目标：`pm25`、`pm10`、`no2`、`co`、`o3`、`so2`、`rain`、`temp`、`press` |
| `mask_ratio` | 0.25 | 每个样本中被遮蔽、作为目标的站点比例 |
| `k_top` | 6 | 每个时间步的邻居数 |
| `window` / `skip_step` | 6 / 4 | 历史时间步数 / 步间隔（小时） |
| `max_step` | 4 | 环形估计的步数 `m` |
| `hid_dim` / `dim_feed` | 64 / 64 | MLP 隐层维度 / Transformer 前馈维度 |
| `dym_graph` | true | 使用邻居聚合（`true`）或 IDW + SES（`false`），需在 `model.json` 中修改 |
| `epochs`、`lr`、`w_decay` | 200、1e-3、1e-5 | Adam 优化器，学习率每 40 个 epoch 减半 |

## 目录结构

```
configs/default.json   数据、训练与 DataLoader 配置
configs/model.json     模型超参数
config.py              合并 JSON 配置与命令行参数
dataset.py             预处理、遮蔽、邻居采样、DataLoader
layers.py              时空编码与通用层
model.py               STFNN：FeatFusion、RingEstimation、STAttention、PyramidalInference
train.py               训练、验证与测试流程
logger.py              日志与模型保存
utils.py               归一化器与旋度（curl）指标
run.sh                 超参数扫描
```

## 引用

```bibtex
@inproceedings{feng2024stfnn,
  title     = {Spatio-Temporal Field Neural Networks for Air Quality Inference},
  author    = {Feng, Yutong and Wang, Qiongyan and Xia, Yutong and Huang, Junlin and Zhong, Siru and Liang, Yuxuan},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, {IJCAI-24}},
  pages     = {7260--7268},
  year      = {2024},
  doi       = {10.24963/ijcai.2024/803}
}
```

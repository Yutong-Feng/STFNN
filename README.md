# STFNN: Spatio-Temporal Field Neural Networks for Air Quality Inference

[中文](README_zh.md)

## 🔥 News

- **[2026-10]** We refactored the code into a cleaner and more concise version: [STFNN-refactor](https://github.com/Yutong-Feng/STFNN-refactor).

Official PyTorch implementation of **"Spatio-Temporal Field Neural Networks for Air Quality Inference"** (IJCAI-24, AI for Good track) [[paper]](https://www.ijcai.org/proceedings/2024/0803.pdf).

Air quality inference estimates pollutant concentrations at locations without monitoring stations. STFNN treats air quality as a continuous **spatio-temporal field** over coordinates `(lng, lat, t)` and learns the field's **gradient** instead of its values. For each target location, a neighbor's observed reading is moved toward the target by integrating the learned gradient along a straight path. The per-neighbor estimates are then combined with learned weights. The paper calls this hybrid of a global field and a local spatio-temporal graph **Pyramidal Inference**:

$$\hat{y}^{tar} = \sum_{i=1}^{K} w_i \Big( y_i^{src} + \sum_{j=1}^{m} \mathbf{D}_{i,j} \cdot \vec{r}_i \Big)$$

| Paper component | Code |
|---|---|
| Spatio-Temporal Encoding: `[lng_z, lat_z]` + sin/cos of day/week/month/year periods | `layers.py:SimpleSTEmbedding` |
| Initial gradient from neighbor features | `model.py:FeatFusion` |
| Ring Estimation: gradient differences `D` over `m` ring steps | `model.py:RingEstimation`, `model.py:PyramidalInference.pyramidal_aggregation` |
| Neighbor Aggregation: weights `W` from a Transformer decoder + softmax | `model.py:STAttention` |
| Ablation: static IDW + SES weights in place of Neighbor Aggregation | `model.py:PyramidalInference.idw_ses` |

## Installation

```bash
pip install torch numpy pandas tqdm
```

Tested with Python 3.13 and PyTorch 2.x on a CUDA GPU; it falls back to CPU when no GPU is available.

## Data

The nationwide Chinese air quality dataset (2018-01-01 to 2018-12-31, hourly, 1085 stations) is a NumPy array `air_2018.npy` with shape `(T, N, F) = (8760, 1085, 16)`:

| Index | Group | Features | Handling |
|---|---|---|---|
| 0–5 | Air quality | PM2.5, PM10, NO2, CO, O3, SO2 | z-score per station (the target column is excluded) |
| 6–10 | Meteorology, continuous | Rainfall, Temperature, Pressure, Humidity, Wind speed | z-score per station |
| 11–12 | Meteorology, categorical | Wind direction, Weather | embedding |
| 13–15 | Temporal context | Time of day, Day of week, is_holiday | embedding |

Station coordinates come from `station_pm25_mean.csv`; only its `longitude` and `latitude` columns are used. Row order must match the station axis of `air_2018.npy`. Put both files in one folder and set `data_folder` in `configs/default.json`.

Preprocessing in `dataset.py:get_air_loader_normalizer`:
- Drops time steps where more than `drop_limit` of the stations miss the target.
- Drops stations with duplicate coordinates.
- Splits the data chronologically into 60% train, 20% validation and 20% test.
- In every sample, masks a random `mask_ratio` of the stations and treats them as targets.
- For each target, collects the `k_top` nearest stations at each of `window + 1` time steps spaced `skip_step` hours apart.

## Usage

Train and evaluate with the default configuration:

```bash
python train.py
```

Any configuration key can be overridden from the command line:

```bash
python train.py --label no2 --mask_ratio 0.5
```

Logs and the best checkpoint (lowest validation MAE) are written to `logs/<timestamp>/training.log` and `logs/<timestamp>/model_parameter.pth`. Test MAE, MAPE and RMSE are reported in original units at the end of training.

### Configuration

Priority: command line > `configs/model.json` > `configs/default.json`.

| Key | Default | Meaning |
|---|---|---|
| `label` | `pm25` | target: `pm25`, `pm10`, `no2`, `co`, `o3`, `so2`, `rain`, `temp`, `press` |
| `mask_ratio` | 0.25 | fraction of stations masked as targets per sample |
| `k_top` | 6 | neighbors per time step |
| `window` / `skip_step` | 6 / 4 | number of past time steps / spacing in hours |
| `max_step` | 4 | ring steps `m` in Ring Estimation |
| `hid_dim` / `dim_feed` | 64 / 64 | MLP hidden size / Transformer feed-forward size |
| `dym_graph` | true | Neighbor Aggregation (`true`) or IDW + SES (`false`); edit in `model.json` |
| `epochs`, `lr`, `w_decay` | 200, 1e-3, 1e-5 | Adam, learning rate halved every 40 epochs |

## Project structure

```
configs/default.json   data, training and loader settings
configs/model.json     model hyperparameters
config.py              merges JSON configs with command-line overrides
dataset.py             preprocessing, masking, neighbor sampling, DataLoaders
layers.py              spatio-temporal encodings and shared layers
model.py               STFNN: FeatFusion, RingEstimation, STAttention, PyramidalInference
train.py               training, validation and test loop
logger.py              logging and checkpoint saving
utils.py               normalizer and curl metric
run.sh                 hyperparameter sweep
```

## Citation

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

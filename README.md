# CoRA: Boosting Time Series Forecasting Foundation Models through Correlation-Aware Adapters

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-blue)](https://pytorch.org/)


### Installation

1. Create virtual environment
    ```shell
    conda create -n "CoRA" python=3.10
    conda activate CoRA
    pip install -r requirements.txt
    ```

### Prepaer Datasets

You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1ZrDotV98JWCSfMaQ94XXd6vh0g27GIrB/view?usp=drive_link). Create a separate folder named `./dataset` 

### Prepare Checkpoints for Foundation Models

Currently we release the **TTM** backbone checkpoint. Download `ttm-research-r2.tar.gz` from [Google Drive](https://drive.google.com/file/d/1Yw6spC_Y6HOIZ85y4L8DgXq9I3rhuPlc/view?usp=sharing), then extract it into `ts_benchmark/baselines/pre_train/checkpoints/`:

```shell
mkdir -p ts_benchmark/baselines/pre_train/checkpoints
tar -xzf ttm-research-r2.tar.gz -C ts_benchmark/baselines/pre_train/checkpoints/
```

After extraction the directory should look like `ts_benchmark/baselines/pre_train/checkpoints/ttm-research-r2/...`.

### Train and evaluate model
- Finetuning the backbone without CoRA:

    ```shell
    python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"batch_size": 32, "dataset": "ETTm2", "freq": "min", "horizon": 96, "is_train": 1, "lr": 0.0001, "norm": true, "num_epochs": 20, "patience": 3, "sampling_rate": 0.05, "seq_len": 512, "target_dim": 7}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTm2/TTM"
    ```

- Finetuning the backbone with CoRA:

    ```shell
    python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"batch_size": 32, "dataset": "ETTm2", "freq": "min", "horizon": 96, "is_train": 1, "lr": 0.001, "norm": true, "num_epochs": 20, "patience": 3, "sampling_rate": 0.05, "seq_len": 512, "target_dim": 7, "train_stages": "plugin,joint"}' --plugin-hyper-params '{"K": 2, "M": 2, "backbone_lr": 0.0001, "beta": 0.2, "de": 2, "dropout": 0.05, "gama": 0.0005, "head_dropout": 0.05, "num_after": 2, "num_before": 1, "plugin_dim": 64, "plugin_lr": 0.0001, "thresold": 0.3}' --adapter "Plugin_adapter" --eval-backend "sequential" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTm2/TTM"
    ```

    
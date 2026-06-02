import copy
import math
import os
from typing import Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import optim

from ts_benchmark.baselines.time_series_library.utils.tools import EarlyStopping
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    get_time_mark,
    train_val_split,
)
from ts_benchmark.models.model_base import BatchMaker, ModelBase
from ts_benchmark.plugin.plugin import Plugin

DEFAULT_LLM_BASED_HYPER_PARAMS = {
    "num_samples": 100,
    "quantiles_num": 20,
    "ckpt_path":"",
    "dataset":"etth1",
    "patience": 3,
    "num_epochs": 10,
    "lradj": "type1",
    "freq": "H",
    "batch_size": 64,
    'label_len':96,
    "num_workers": 0,
    "freq": "h",
    "sampling_rate": 0.05,
    "sampling_strategy": "uniform",
    "sampling_basis": "sample",
    "is_train": 0,
    "get_train": 0,
    "lr": 0.0001,
    "get_pt": 0,
    "is_gpt": 1, # autotimes
    "patch_size": 16,
    "kernel_size": 25,
    "pretrain": 1,
    "seq_len": 96,
    "horizon": 96,
    "stride": 8,
    "gpt_layers": 3,
    "d_model": 768,
    "freeze":1,

    "use_multi_gpu": 0, # gpt4ts
    "local_rank": 0,
    "mix_embeds": 1,
    "mlp_hidden_layers": 0,
    "mlp_hidden_dim": 256,
    "mlp_activation": 'tanh',
    "dropout": 0.1,
    "alpha": 0.5,
    "token_len": 96,
    "use_p": 0,
    "loss":"MSE",
    "ln": 0, # s2lpllm
    "task_name": "long_term_forecast",
    "patch_size": 16,
    "stride": 8,
    "pretrained": 1,
    "pool_size": 1000,
    "prompt_length": 8,
    "trend_length": 96,
    "seasonal_length": 24,
    "prompt_init": 'text_prototype',

    "d_ff": 32, #timellm
    "llm_dim": 768,
    "patch_len": 16,
    "llm_model": "GPT2",
    "prompt_domain": 1,
    "n_heads": 8,
    "enc_in": 7,
    "llm_layers": 12,
    "content": "",

    "max_token_num": 50, # unitime
    "mask_rate": 0.5,
    "max_backcast_len": 96,
    "max_forecast_len": 720,
    "ts_embed_dropout": 0.3,
    "lm_ft_type": "fpt",
    "instruct_path": "dataset/instruct.json",
    "lm_layer_num": 6,
    "dec_trans_layer_num": 2, 
    "ts_embed_dropout": 0.3,
    "dec_head_dropout": 0.1,

    "top_k" : 5, #LLMMixer
    "num_kernels": 6,
    # "d_model": 16,
    # "n_heads": 4,
    "e_layers": 2,
    "d_layers": 1,
    "moving_avg": 25,
    "factor": 1,
    "distil": 1,
    "channel_independence": 1,
    "decomp_method": "moving_avg",
    "down_sampling_layers": 3,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",
    "use_future_temporal_feature": 0,
    "llm_path": "ts_benchmark/baselines/LLM/checkpoints/roberta-base",
    "tokenizer_path": "ts_benchmark/baselines/LLM/checkpoints/roberta-base",
    "embed": "timeF",
    "description": "",
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "word_embedding_path": "ts_benchmark/baselines/LLM/checkpoints/wte_pca_500.pt",
}

DEFAULT_LLM_PLUGIN_HYPER_PARAMS = {
    "backbone_lr": 0.00005,
    "plugin_lr": 0.0001,
    "backbone_lradj": "cosine",
    "plugin_lradj": "type1",
    "beta": 0.2,
    "dropout": 0.1,
    "head_dropout": 0.1,
    "num_after": 2,
    "num_before": 2,
    "plugin_dim": 128,
    "gama": 0.001,
    "K": 2,
    "M": None,
    "de": 2,
    "thresold": 0.2,
}


class LLMConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_LLM_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.horizon


class PluginConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_LLM_PLUGIN_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.horizon


class LLMPluginAdapter(ModelBase):
    def __init__(self, model_name, model_class, **kwargs):
        super().__init__()
        self.config = LLMConfig(**kwargs)
        self.plugin_config = PluginConfig(**self.config.plugin)
        self.config.plugin = self.plugin_config
        self._model_name = model_name
        self.model_class = model_class
        self.scaler = StandardScaler()
        self.seq_len = self.config.seq_len
        self.ending = False

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    @property
    def model_name(self):
        return self._model_name

    def _forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq is None:
            self.config.freq = self.config.freq.lower()
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = self.config.seq_len // 2

    def _finalize_plugin_hyper_params(self):
        if getattr(self.config.plugin, "M", None) is None:
            self.config.plugin.M = self.config.enc_in // 10 + 1

    def _padding_time_stamp_mark(self, time_stamps_list: np.ndarray, padding_len: int):
        padding_time_stamp = []
        for time_stamps in time_stamps_list:
            start = time_stamps[-1]
            expand_time_stamp = pd.date_range(
                start=start,
                periods=padding_len + 1,
                freq=self.config.freq.upper(),
            )
            padding_time_stamp.append(expand_time_stamp.to_numpy()[-padding_len:])
        padding_time_stamp = np.stack(padding_time_stamp)
        whole_time_stamp = np.concatenate((time_stamps_list, padding_time_stamp), axis=1)
        return get_time_mark(whole_time_stamp, 1, self.config.freq)

    def _get_plugin_parameters(self):
        return [
            param
            for name, param in self.model.named_parameters()
            if not name.startswith("fm.")
        ]

    def _get_backbone_parameters(self):
        return list(self.model.fm.parameters())

    def _count_trainable_parameters(self, parameters):
        return sum(param.numel() for param in parameters if param.requires_grad)

    def _compute_stage_lr(self, base_lr, lradj, epoch):
        if lradj == "type1":
            return base_lr * (0.5 ** (epoch - 1))
        if lradj == "cosine":
            return base_lr / 2 * (1 + math.cos(epoch / self.config.num_epochs * math.pi))
        return None

    def _adjust_stage_learning_rates(self, optimizer, epoch, lr_settings):
        for param_group in optimizer.param_groups:
            group_name = param_group.get("name", "default")
            base_lr, lradj = lr_settings[group_name]
            lr = self._compute_stage_lr(base_lr, lradj, epoch)
            if lr is not None:
                param_group["lr"] = lr

    def _build_stage_optimizer(self, stage_name):
        plugin_params = self._get_plugin_parameters()
        backbone_params = self._get_backbone_parameters()

        if stage_name == "plugin":
            self.model.freeze_backbone()
            optimizer = optim.Adam(
                [param for param in plugin_params if param.requires_grad],
                lr=self.config.plugin.plugin_lr,
            )
            lr_settings = {
                "default": (
                    self.config.plugin.plugin_lr,
                    self.config.plugin.plugin_lradj,
                )
            }
        elif stage_name == "joint":
            self.model.unfreeze_backbone()
            optimizer = optim.Adam(
                [
                    {
                        "params": [param for param in plugin_params if param.requires_grad],
                        "lr": self.config.plugin.plugin_lr,
                        "name": "plugin",
                    },
                    {
                        "params": [param for param in backbone_params if param.requires_grad],
                        "lr": self.config.plugin.backbone_lr,
                        "name": "backbone",
                    },
                ]
            )
            lr_settings = {
                "plugin": (
                    self.config.plugin.plugin_lr,
                    self.config.plugin.plugin_lradj,
                ),
                "backbone": (
                    self.config.plugin.backbone_lr,
                    self.config.plugin.backbone_lradj,
                ),
            }
        else:
            raise ValueError(f"Unsupported training stage: {stage_name}")

        trainable_counts = {
            "plugin": self._count_trainable_parameters(plugin_params),
            "backbone": self._count_trainable_parameters(backbone_params),
        }
        return optimizer, lr_settings, trainable_counts

    def validate(self, valid_data_loader, criterion):
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                dec_input = torch.zeros_like(target[:, -self.config.horizon :, :]).float()
                dec_input = torch.cat(
                    [target[:, : self.config.label_len, :], dec_input], dim=1
                ).to(device)

                output = self.model(input, dec_input, input_mark, target_mark, device)
                target = target[:, -self.config.horizon :, :]
                output = output[:, -self.config.horizon :, :]
                total_loss.append(criterion(output, target).detach().cpu().numpy())

        self.model.train()
        return np.mean(total_loss)

    def forecast_fit(
        self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float
    ) -> "ModelBase":
        self._forecasting_hyper_param_tune(train_valid_data)
        self._finalize_plugin_hyper_params()

        print("---------------------------------------------------------- CoRA + ", self.model_name)
        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )

        self.scaler.fit(train_data.values)
        if config.norm:
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns,
                index=train_data.index,
            )

        if train_ratio_in_tv != 1:
            if config.norm:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )
            _, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
                data_info="val",
            )

        _, train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_valid_data.shape[1] > 1,
            data_info="train",
            sampling_rate=config.sampling_rate,
            sampling_strategy=config.sampling_strategy,
            sampling_basis=config.sampling_basis,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = self.model_class(self.config, device)
        self.model = Plugin(backbone=backbone, configs=self.config).to(device)

        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")

        criterion = nn.MSELoss()
        for stage_name, stage_title in [
            ("plugin", "plugin only"),
            ("joint", "joint finetune"),
        ]:
            if (
                stage_name == "joint"
                and hasattr(self, "early_stopping")
                and self.early_stopping.check_point is not None
            ):
                self.model.load_state_dict(self.early_stopping.check_point)

            optimizer, lr_settings, trainable_counts = self._build_stage_optimizer(
                stage_name
            )
            self.early_stopping = EarlyStopping(patience=config.patience)
            self.ending = True
            print(f"---------------- Training stage: {stage_title}")
            print(
                "Trainable parameters "
                f"(plugin={trainable_counts['plugin']}, backbone={trainable_counts['backbone']})"
            )

            for epoch in range(config.num_epochs):
                self.model.train()
                for input, target, input_mark, target_mark in train_data_loader:
                    input, target, input_mark, target_mark = (
                        input.to(device),
                        target.to(device),
                        input_mark.to(device),
                        target_mark.to(device),
                    )
                    dec_input = torch.zeros_like(target[:, -config.horizon :, :]).float()
                    dec_input = torch.cat(
                        [target[:, : config.label_len, :], dec_input], dim=1
                    ).to(device)

                    output, loss_cc = self.model(
                        input, dec_input, input_mark, target_mark, device
                    )
                    target = target[:, -config.horizon :, :]
                    output = output[:, -config.horizon :, :]
                    loss = (
                        criterion(output, target)
                        + (output - target).abs().mean() * self.config.alpha
                        + loss_cc.mean()
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if train_ratio_in_tv != 1:
                    valid_loss = self.validate(valid_data_loader, criterion)
                    self.early_stopping(valid_loss, self.model)
                    if self.early_stopping.early_stop:
                        self.ending = False
                        print(f"Early Stopping {stage_title} ----------------")
                        break
                else:
                    self.early_stopping.check_point = copy.deepcopy(
                        self.model.state_dict()
                    )

                self._adjust_stage_learning_rates(optimizer, epoch + 1, lr_settings)

            if self.early_stopping.check_point is not None:
                self.model.load_state_dict(self.early_stopping.check_point)

        os.makedirs("ts_benchmark/baselines/LLM/checkpoints/LLM", exist_ok=True)

    def forecast(self, horizon: int, train: pd.DataFrame) -> np.ndarray:
        return None

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        if hasattr(self, "early_stopping") and self.early_stopping.check_point is not None:
            self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = input_data["input"]

        if self.config.norm:
            origin_shape = input_np.shape
            flattened_data = input_np.reshape((-1, input_np.shape[-1]))
            input_np = self.scaler.transform(flattened_data).reshape(origin_shape)

        input_index = input_data["time_stamps"]
        padding_len = (math.ceil(horizon / self.config.horizon) + 1) * self.config.horizon
        all_mark = self._padding_time_stamp_mark(input_index, padding_len)

        answers = self._perform_rolling_predictions(horizon, input_np, all_mark, device)

        if self.config.norm:
            flattened_data = answers.reshape((-1, answers.shape[-1]))
            answers = self.scaler.inverse_transform(flattened_data).reshape(
                answers.shape
            )

        return answers

    def _perform_rolling_predictions(
        self,
        horizon: int,
        input_np: np.ndarray,
        all_mark: np.ndarray,
        device: torch.device,
    ) -> list:
        rolling_time = 0
        input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
            input_np, None, all_mark, rolling_time
        )
        with torch.no_grad():
            answers = []
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input, dec_input, input_mark, target_mark = (
                    torch.tensor(input_np, dtype=torch.float32).to(device),
                    torch.tensor(target_np, dtype=torch.float32).to(device),
                    torch.tensor(input_mark_np, dtype=torch.float32).to(device),
                    torch.tensor(target_mark_np, dtype=torch.float32).to(device),
                )
                output = self.model(input, dec_input, input_mark, target_mark, device)
                column_num = output.shape[-1]
                real_batch_size = output.shape[0]
                answer = (
                    output.cpu()
                    .numpy()
                    .reshape(real_batch_size, -1, column_num)[
                        :, -self.config.horizon :, :
                    ]
                )
                answers.append(answer)
                if sum(a.shape[1] for a in answers) >= horizon:
                    break
                rolling_time += 1
                output = output.cpu().numpy()[:, -self.config.horizon :, :]
                input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
                    input_np, output, all_mark, rolling_time
                )

        answers = np.concatenate(answers, axis=1)
        return answers[:, -horizon:, :]

    def _get_rolling_data(
        self,
        input_np: np.ndarray,
        output: Optional[np.ndarray],
        all_mark: np.ndarray,
        rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if rolling_time > 0:
            input_np = np.concatenate((input_np, output), axis=1)
            input_np = input_np[:, -self.config.seq_len :, :]
        target_np = np.zeros(
            (
                input_np.shape[0],
                self.config.label_len + self.config.horizon,
                input_np.shape[2],
            )
        )
        target_np[:, : self.config.label_len, :] = input_np[
            :, -self.config.label_len :, :
        ]
        advance_len = rolling_time * self.config.horizon
        input_mark_np = all_mark[:, advance_len : self.config.seq_len + advance_len, :]
        start = self.config.seq_len - self.config.label_len + advance_len
        end = self.config.seq_len + self.config.horizon + advance_len
        target_mark_np = all_mark[:, start:end, :]
        return input_np, target_np, input_mark_np, target_mark_np


def generate_model_factory(
    model_name: str, model_class: type, required_args: dict
) -> Dict:
    def model_factory(**kwargs) -> LLMPluginAdapter:
        return LLMPluginAdapter(model_name, model_class, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": required_args,
    }


def LLM_Plugin_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")

    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        },
    )

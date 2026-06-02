from pathlib import Path
import pandas as pd
import torch
import yaml
from torch import nn
import torch.nn.functional as F
from huggingface_hub import HfApi, snapshot_download
from ts_benchmark.baselines.pre_train.submodules.TinyTimeMixer.modeling_tinytimemixer import TinyTimeMixerForPrediction
from pandas.tseries.frequencies import to_offset

TTM_CHECKPOINT_DIR = Path("ts_benchmark/baselines/pre_train/checkpoints/ttm-research-r2")

DEFAULT_FREQUENCY_MAPPING = {
    "oov": 0,
    "min": 4,  # minutely
    "2min": 2,
    "5min": 3,
    "10min": 4,
    "15min": 5,
    "30min": 6,
    "h": 7,  # hourly
    "d": 8,  # daily
    "W": 9,  # weekly
}

class TinyTimeMixer(nn.Module):

    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.target_dim = config.target_dim
        self.freq = config.freq
        self.dataset = config.dataset
        self.frequency_mapping = DEFAULT_FREQUENCY_MAPPING
        self.token = self.get_frequency_token(self.freq)
        self.model_id = "ibm-research/ttm-research-r2"
        ttm_model_revision = self.resolve_revision()
        ttm_model_path = self.ensure_local_checkpoint(ttm_model_revision)

        self.model = TinyTimeMixerForPrediction.from_pretrained(
            str(ttm_model_path),
            local_files_only=True,
        )
        self.loc = None
        self.scale = None

    def ensure_local_checkpoint(self, revision):
        local_path = TTM_CHECKPOINT_DIR / self._local_checkpoint_name(revision)

        if self._checkpoint_complete(local_path):
            return local_path

        local_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.model_id,
            revision=revision,
            local_dir=str(local_path),
        )
        return local_path

    @staticmethod
    def _safe_revision_name(revision):
        return revision.replace("/", "__")

    def _local_checkpoint_name(self, revision):
        if revision == "main":
            return f"{self.context_length}-{self.prediction_length}-ft-r2"
        return self._safe_revision_name(revision)

    @staticmethod
    def _checkpoint_complete(local_path):
        required_files = ("config.json",)
        has_model_file = any(
            (local_path / filename).exists()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        return all((local_path / filename).exists() for filename in required_files) and has_model_file

    def resolve_revision(self):
        config_path = Path("ts_benchmark/baselines/pre_train/checkpoints/ttm.yaml")
        if config_path.exists():
            with config_path.open("r") as file:
                model_revisions = yaml.safe_load(file)
            revision = model_revisions["research-use-models"].get(
                f"r2-{self.context_length}-{self.prediction_length}-freq",
                {},
            ).get("revision")
            if revision:
                return revision

        candidate_revision = f"{self.context_length}-{self.prediction_length}-ft-r2"
        local_candidate_path = TTM_CHECKPOINT_DIR / self._safe_revision_name(candidate_revision)
        if self._checkpoint_complete(local_candidate_path):
            return candidate_revision

        try:
            branches = {
                branch.name
                for branch in HfApi().list_repo_refs(self.model_id).branches
            }
        except Exception:
            branches = set()

        if candidate_revision in branches:
            return candidate_revision

        local_main_path = TTM_CHECKPOINT_DIR / self._local_checkpoint_name("main")
        if self._checkpoint_complete(local_main_path):
            return "main"

        return "main"

        
    def get_frequency_token(self, token_name: str):
        token = self.frequency_mapping.get(token_name, None)
        if token is not None:
            return token

        # try to map as a frequency string
        try:
            token_name_offs = to_offset(token_name).freqstr
            token = self.frequency_mapping.get(token_name_offs, None)
            if token is not None:
                return token
        except ValueError:
            # lastly try to map the timedelta to a frequency string
            token_name_td = pd._libs.tslibs.timedeltas.Timedelta(token_name)
            token_name_offs = to_offset(token_name_td).freqstr
            token = self.frequency_mapping.get(token_name_offs, None)
            if token is not None:
                return token

        token = self.frequency_mapping["oov"]

        return token

    def _get_freq_token(self, batch_size: int, device: torch.device):
        return torch.full((batch_size,), self.token, device=device, dtype=torch.long)
    
    def forward(self, inputs, dec_inp, x_mark_enc, x_mark_dec, device=None, num_samples=None): 
        B, C, K = inputs.shape
        freq_token = self._get_freq_token(B, inputs.device if device is None else device)
        outputs = self.model(past_values=inputs, freq_token=freq_token)
        self.loc = outputs.loc
        self.scale = outputs.scale
        point_forecast = outputs.prediction_outputs
        return point_forecast

    def get_settings(self):
        config = self.model.config
        return (
            config.d_model,
            config.patch_length,
            config.patch_stride,
            config.num_patches,
        )

    def forcast_for_plugin(self, inputs, x_mark_enc, dec_inp, x_mark_dec, device=None):
        batch_size = inputs.shape[0]
        freq_token = self._get_freq_token(
            batch_size, inputs.device if device is None else device
        )
        outputs = self.model(
            past_values=inputs,
            freq_token=freq_token,
            output_hidden_states=False,
            return_loss=False,
        )
        self.loc = outputs.loc
        self.scale = outputs.scale
        return outputs.prediction_outputs, outputs.backbone_hidden_state

    def patchify_for_plugin(self, inputs):
        observed_mask = torch.ones_like(inputs)
        scaled_inputs, _, _ = self.model.backbone.scaler(inputs, observed_mask)
        return self.model.backbone.patching(scaled_inputs)

    def denorm_for_plugin(self, inputs):
        if self.loc is None or self.scale is None:
            raise RuntimeError('TinyTimeMixer denormalization state is not initialized.')
        return inputs * self.scale + self.loc

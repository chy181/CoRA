import torch
from torch import nn
from einops import rearrange, repeat
from pathlib import Path
from ts_benchmark.baselines.pre_train.submodules.Timer.models.Timer import Model as Timer


def _convert_timer_state_dict(state_dict):
    converted = {}
    for key, value in state_dict.items():
        if key == "embedding.weight":
            converted["patch_embedding.value_embedding.weight"] = value
        elif key == "embedding.bias":
            continue
        elif key.startswith("blocks."):
            if ".attention.inner_attention.attn_bias." in key:
                continue
            converted["decoder." + key[len("blocks."):]] = value
        elif key.startswith("head."):
            converted["proj." + key[len("head."):]] = value
        else:
            converted[key] = value
    return converted


def _load_timer_checkpoint(model):
    checkpoint_dir = Path("ts_benchmark/baselines/pre_train/checkpoints/timer")
    state_path = checkpoint_dir / "Timer_67M_UTSD_4G.pth"
    jit_path = checkpoint_dir / "Timer_67M_UTSD_4G.pt"

    if state_path.exists():
        state_dict = torch.load(state_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "embedding.weight" in state_dict:
            state_dict = _convert_timer_state_dict(state_dict)
        missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
        missing = [key for key in missing if key != "patch_embedding.position_embedding.pe"]
        if missing or unexpected:
            raise RuntimeError(
                f"Timer checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        return

    jit_model = torch.jit.load(str(jit_path))
    model.load_state_dict(jit_model.state_dict())

class Config():
    def __init__(self, config) -> None:
        # self.ckpt_path = 'ts_benchmark/baselines/pre_train/checkpoints/timer/Timer_forecast_1.0.ckpt'
        # self.ckpt_path = 'ts_benchmark/baselines/pre_train/checkpoints/timer/Timer_67M_UTSD_4G.pt'
        self.ckpt_path = ''

        self.task_name = 'forecast'
        self.d_model = 1024
        self.d_ff = 2048
        self.e_layers = 8
        self.n_heads = 8
        self.factor = 3
        self.dropout = 0.1
        self.activation = 'gelu'
        self.patch_len = 96
        self.output_attention = None
        self.use_plugin = False

class TimerModel(nn.Module):
# class Timer(nn.Module):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__()
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.target_dim = config.target_dim
        self.freq = config.freq
        self.dataset = config.dataset
        self.config = config

        self.output_patch_len = 96 # fixed by the pre-trained model
        self.label_len = config.seq_len - 96
        
        self.model =Timer(Config(config))
        _load_timer_checkpoint(self.model)
        # self.model = jit_model
    def forward(self, inputs, dec_inp, x_mark_enc, x_mark_dec, device=None, num_samples=None):        
        
        B, _, K = inputs.shape

        inputs = rearrange(inputs, 'b l k -> (b k) l 1')
        x_mark_enc = repeat(x_mark_enc, 'b l f -> (b k) l f', k=K)
        x_mark_dec = repeat(x_mark_dec, 'b l f -> (b k) l f', k=K)

        dec_inp = torch.zeros_like(inputs[:, -self.prediction_length:, :]).float()
        dec_inp = torch.cat((inputs[:, -self.label_len:, ...], dec_inp), dim=1).float()

        if self.config.is_train == 0:
            inference_steps = self.prediction_length // self.output_patch_len
            dis = self.prediction_length - inference_steps * self.output_patch_len
            if dis != 0:
                inference_steps += 1

            pred_y = []

            for j in range(inference_steps):
                if len(pred_y) != 0:
                    inputs = torch.cat([inputs[:, self.output_patch_len:, :], pred_y[-1]], dim=1)
                    tmp = x_mark_dec[:, j - 1:j, :]
                    x_mark_enc = torch.cat([x_mark_enc[:, 1:, :], tmp], dim=1)

                outputs = self.model(inputs, x_mark_enc, dec_inp, x_mark_dec)
                pred_y.append(outputs[:, -self.output_patch_len:, :])

            pred_y = torch.cat(pred_y, dim=1)
            if dis != 0:
                pred_y = pred_y[:, :-dis, :]
            pred_y = rearrange(pred_y, '(b k) l 1 -> b l k', b=B, k=K)
            pred_y = pred_y[:, :self.prediction_length, :]
        else:
            outputs = self.model(inputs, x_mark_enc, dec_inp, x_mark_dec)
            pred_y = rearrange(outputs, '(b k) l 1 -> b l k', b=B, k=K)
            
        return pred_y
    
    def get_settings(self,):
        return self.model.d_model, self.model.patch_len, self.model.stride, (self.config.seq_len - self.model.patch_len)//self.model.patch_len + 1
    
    def forcast_for_plugin(self,inputs, x_mark_enc, dec_inp, x_mark_dec,device):
        B, _, K = inputs.shape
        inputs = rearrange(inputs, 'b l k -> (b k) l 1')
        outputs, embedding = self.model.forcast_for_plugin(inputs, x_mark_enc, dec_inp, x_mark_dec)
        outputs = rearrange(outputs, '(b k) l 1 -> b l k', b=B, k=K)
        embedding = rearrange(embedding,'(b k) p d -> b k p d', b=B, k=K)

        # outputs = outputs[:, -self.config.seq_len:, :]
        return outputs,embedding
    
    def forcast_for_plugin_decoder(self,inputs, x_mark_enc, dec_inp, x_mark_dec):
        B, _, K = inputs.shape
        inputs = rearrange(inputs, 'b l k -> (b k) l 1')
        outputs, embedding = self.model.forcast_for_plugin_decoder(inputs, x_mark_enc, dec_inp, x_mark_dec)
        outputs = rearrange(outputs, '(b k) l 1 -> b l k', b=B, k=K)
        embedding = rearrange(embedding,'(b k) p d -> b k p d', b=B, k=K)

        # outputs = outputs[:, -self.config.seq_len:, :]
        return outputs,embedding
    
    def forcast_for_plugin_out(self,inputs, x_mark_enc, dec_inp, x_mark_dec):
        B, _, K = inputs.shape
        inputs = rearrange(inputs, 'b l k -> (b k) l 1')
        embedding = self.model.forcast_for_plugin_out(inputs, x_mark_enc, dec_inp, x_mark_dec)
        embedding = rearrange(embedding,'(b k) p d -> b k p d', b=B, k=K)

        return embedding
    
    def forcast_for_plugin_in(self, embedding):
        
        B, K, _, _ = embedding.shape
        outputs = self.model.forcast_for_plugin_in(embedding)
        outputs = rearrange(outputs, '(b k) l 1 -> b l k', b=B, k=K)

        return outputs
    

    def denorm_for_plugin(self,inputs):
        B, _, K = inputs.shape
        inputs = rearrange(inputs, 'b l k -> (b k) l 1')
        outputs = self.model.denorm_for_plugin(inputs)
        outputs = rearrange(outputs, '(b k) l 1 -> b l k', b=B, k=K)
        return outputs

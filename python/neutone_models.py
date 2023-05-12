import logging
import os
import pathlib
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import torch as tr
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


# Model for benchmarking that lets the audio pass through unchanged
class PassThrough(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


# DSP clipper
class Clipper(nn.Module):
    def forward(self, x: Tensor, min_val: Tensor, max_val: Tensor, gain: Tensor) -> Tensor:
        tr.neg(min_val, out=min_val)
        tr.mul(gain, min_val, out=min_val)
        tr.mul(gain, max_val, out=max_val)
        tr.clip(x, min=min_val, max=max_val, out=x)
        return x


# 1-layer MLP model
class MLP(nn.Module):
    def __init__(self, n_hidden: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x.unsqueeze(-1))
        return x.squeeze(-1)


# 2-layer MLP model
class MLPx2(nn.Module):
    def __init__(self, n_hidden: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x.unsqueeze(-1))
        return x.squeeze(-1)


# 1-layer CNN model
class CNN1D(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 2, n_hidden: int = 16) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, n_hidden, (3,), padding="same"),
            nn.ReLU(),
            nn.Conv1d(n_hidden, out_ch, (1,)),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x.unsqueeze(0))
        return x.squeeze(0)


# 2-layer CNN model
class CNN1Dx2(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 2, n_hidden: int = 16) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, n_hidden, (3,), padding="same"),
            nn.ReLU(),
            nn.Conv1d(n_hidden, n_hidden, (3,), padding="same"),
            nn.ReLU(),
            nn.Conv1d(n_hidden, out_ch, (1,)),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x.unsqueeze(0))
        return x.squeeze(0)


# Minimum possible LSTM model with no state saving
class LSTMMinimal(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 1) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.lstm = nn.LSTM(in_ch, 1, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        lstm_in = x.unsqueeze(-1)
        lstm_out, new_hidden = self.lstm(lstm_in)
        y_hat = lstm_out.squeeze(-1)
        return y_hat


# Base class for hidden state models like LSTMs
class HiddenStateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Must be initialized as a tensor for torchscript tracing
        self.hidden: Tuple[Tensor, Tensor] = (tr.zeros((1,)), tr.zeros((1,)))
        self.is_hidden_init = False

    def update_hidden(self, hidden: Tuple[Tensor, Tensor]) -> None:
        self.hidden = hidden
        self.is_hidden_init = True

    def detach_hidden(self) -> None:
        if self.is_hidden_init:
            # TODO(cm): check whether clone is required or not
            self.hidden = tuple((h.detach() for h in self.hidden))

    def clear_hidden(self) -> None:
        self.is_hidden_init = False


# Functional LSTM model with conditioning signal
class LSTMEffectModel(HiddenStateModel):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 1,
                 n_hidden: int = 64,
                 latent_dim: int = 1) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_hidden = n_hidden
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(in_ch + latent_dim, n_hidden, batch_first=True)
        self.fc = nn.Linear(n_hidden, out_ch)

    def forward(self, x: Tensor, latent: Tensor) -> Tensor:
        x = x.unsqueeze(0)
        latent = latent.view(1, 1, -1)
        lstm_in = tr.cat([latent, x], dim=1)
        lstm_in = tr.swapaxes(lstm_in, 1, 2)
        if self.is_hidden_init:
            lstm_out, new_hidden = self.lstm(lstm_in, self.hidden)
        else:
            lstm_out, new_hidden = self.lstm(lstm_in)
        fc_out = self.fc(lstm_out)
        fc_out = tr.swapaxes(fc_out, 1, 2)
        y_hat = fc_out + x
        y_hat = tr.tanh(y_hat)
        self.update_hidden(new_hidden)
        return y_hat.squeeze(0)


# Wrapper for exporting a model to Neutone
class BenchmarkModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "benchmark"

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "Benchmark model."

    def get_model_long_description(self) -> str:
        return "Benchmark model."

    def get_technical_description(self) -> str:
        return "Benchmark model."

    def get_tags(self) -> List[str]:
        return ["benchmark"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("min", "min clip threshold", default_value=0.15),
            NeutoneParameter("max", "max clip threshold", default_value=0.15),
            NeutoneParameter("gain", "scale clip threshold", default_value=1.0),
        ]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return False
        # return True

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return False
        # return True

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        # return [44100]
        # return [48000]
        return []  # Supports all sample rates

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        # return [128]
        # return [512]
        # return [727]
        # return [2048]
        return []  # Supports all buffer sizes

    def get_look_behind_samples(self) -> int:
        # return 512
        return 0

    # Comment this method to perform mean aggregation per block
    def aggregate_params(self, params: Tensor) -> Tensor:
        return params

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        min_val, max_val, gain = params["min"], params["max"], params["gain"]
        x = self.model.forward(x)  # PassThrough, MLP, CNN, LSTMMinimal
        # x = self.model.forward(x, min_val, max_val, gain)  # Clipper
        # x = self.model.forward(x, min_val)  # LSTMEffectModel
        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default="export_model")
    args = parser.parse_args()
    root_dir = pathlib.Path(args.output)

    # Select model to benchmark
    model = PassThrough()
    # model = Clipper()
    # model = MLP(n_hidden=16)
    # model = MLP(n_hidden=64)
    # model = MLPx2(n_hidden=16)
    # model = CNN1D(n_hidden=16)
    # model = CNN1Dx2(n_hidden=16)
    # model = LSTMMinimal()
    # model = LSTMEffectModel(in_ch=1, out_ch=1, n_hidden=16)

    # Export Neutone model
    wrapper = BenchmarkModelWrapper(model)
    save_neutone_model(wrapper, root_dir, dump_samples=False, submission=True)

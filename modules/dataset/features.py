import torch
import torch.nn as nn
import torchaudio
from modules.configs.config import Config

class LogMelFeatureExtractor(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        f_min=20,
        f_max=7600,
    ):
        super().__init__()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=Config.NUM_MELS,
            window_fn=torch.hamming_window,
            power=2.0,
            normalized=False,
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power"
        )

    def forward(self, waveform):

        # Mel Spectrogram
        mel = self.mel_transform(waveform)

        # Convert to Log Scale
        mel = self.amplitude_to_db(mel)

        mel = mel.squeeze(0)

        mel = mel - mel.mean(dim=1, keepdim=True)

        return mel
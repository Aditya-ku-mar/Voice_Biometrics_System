import torch
import torch.nn as nn
import torchaudio


class LogMelFeatureExtractor(nn.Module):
    """
    Extract 80-dimensional Log-Mel Filter Bank features.

    Input:
        waveform : (1, N)

    Output:
        features : (80, Time)
    """

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
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=torch.hamming_window,
            power=2.0,
            normalized=False,
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power"
        )

    def forward(self, waveform):
        """
        waveform shape
            (1, N)

        returns
            (80, Time)
        """

        # -----------------------------
        # Mel Spectrogram
        # -----------------------------
        mel = self.mel_transform(waveform)

        # (1,80,T)

        # -----------------------------
        # Convert to Log Scale
        # -----------------------------
        mel = self.amplitude_to_db(mel)

        # -----------------------------
        # Remove Channel Dimension
        # -----------------------------
        mel = mel.squeeze(0)

        # (80,T)

        # -----------------------------
        # Mean Normalization
        # -----------------------------
        mel = mel - mel.mean(dim=1, keepdim=True)

        return mel
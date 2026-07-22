import random

import torch
import torch.nn.functional as F
import torchaudio


class AudioAugmentation:
    """
    Research-style audio augmentation.

    Operations:
        1. Background Noise
        2. Random Gain
        3. Speed Perturbation

    Future extensions:
        - Reverberation
        - MUSAN Noise
        - RIR Simulation
    """

    def __init__(
        self,
        noise_probability=0.30,
        gain_probability=0.30,
        speed_probability=0.30,
    ):

        self.noise_probability = noise_probability
        self.gain_probability = gain_probability
        self.speed_probability = speed_probability

    # --------------------------------------------------

    def add_noise(self, waveform):

        noise = torch.randn_like(waveform)

        snr_db = random.uniform(10, 30)

        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()

        scale = torch.sqrt(
            signal_power /
            (10 ** (snr_db / 10) * noise_power + 1e-10)
        )

        noise = noise * scale

        return waveform + noise

    # --------------------------------------------------

    def random_gain(self, waveform):

        gain_db = random.uniform(-6, 6)

        gain = 10 ** (gain_db / 20)

        return waveform * gain

    # --------------------------------------------------

    def speed_perturb(self, waveform):

        factor = random.choice([0.90, 1.00, 1.10])

        if factor == 1.0:
            return waveform

        sample_rate = 16000
        new_sr = int(sample_rate * factor)

        # Change playback speed
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=new_sr
        )

        # Resample back to 16 kHz
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=new_sr,
            new_freq=sample_rate
        )

        target_length = 3 * sample_rate  # 48000 samples

        if waveform.size(1) > target_length:
            waveform = waveform[:, :target_length]

        elif waveform.size(1) < target_length:
            pad = target_length - waveform.size(1)

            waveform = F.pad(
                waveform,
                (0, pad)
            )

        return waveform

    # --------------------------------------------------

    def __call__(self, waveform):

        # Speed perturbation
        if random.random() < self.speed_probability:
            waveform = self.speed_perturb(waveform)

        # Random gain
        if random.random() < self.gain_probability:
            waveform = self.random_gain(waveform)

        # Add background noise
        if random.random() < self.noise_probability:
            waveform = self.add_noise(waveform)

        return waveform.contiguous()
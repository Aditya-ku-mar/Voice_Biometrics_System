import random

import torch
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

        factor = random.choice(
            [0.90, 1.00, 1.10]
        )

        if factor == 1.0:
            return waveform

        effect = [
            ["speed", str(factor)],
            ["rate", "16000"]
        ]

        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform,
            16000,
            effect
        )

        return waveform

    # --------------------------------------------------

    def __call__(self, waveform):

        if random.random() < self.noise_probability:
            waveform = self.add_noise(waveform)

        if random.random() < self.gain_probability:
            waveform = self.random_gain(waveform)

        if random.random() < self.speed_probability:
            waveform = self.speed_perturb(waveform)

        return waveform
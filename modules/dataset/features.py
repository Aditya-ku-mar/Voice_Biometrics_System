import torch
import torchaudio
from modules.configs.config import Config

class LogMelFeatureExtractor:
    def __init__(self):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=400,                
            win_length=400,
            hop_length=160,            
            n_mels=Config.NUM_MELS,
            power=2.0
        )
        
    def __call__(self, waveform):
        mel = self.mel(waveform)
        mel = torch.clamp(mel, min=1e-6)
        log_mel = torch.log(mel)

        return log_mel.squeeze(0)


import os
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from modules.configs.config import Config
from modules.dataset.features import LogMelFeatureExtractor
from modules.dataset.augmentation import AudioAugmentation

class SpeakerDataset(Dataset):

    def __init__(
        self,
        root_dir,
        sample_rate=Config.SAMPLE_RATE,
        segment_length=Config.SEGMENT_LENGTH,
        train=True,
        augmentation=True
    ):

        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(sample_rate * segment_length)

        self.train = train
        self.use_augmentation = augmentation

        self.feature_extractor = LogMelFeatureExtractor(
            sample_rate=self.sample_rate
        )

        self.augmentation = AudioAugmentation()
        
        self.audio_paths = []
        self.labels = []
        self.speaker_to_index = {}
        self.resamplers = {}
        self._load_dataset()


    def _load_dataset(self):

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"Dataset path not found:\n{self.root_dir}"
            )

        print("=" * 60)
        print("Scanning Dataset...")
        print(self.root_dir)
        print("=" * 60)

        speaker_idx = 0
        speakers = sorted(
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        )

        for speaker in speakers:
            speaker_dir = os.path.join(
                self.root_dir,
                speaker
            )

            speaker_audio = []
            for root, _, files in os.walk(speaker_dir):
                for file in files:
                    if file.lower().endswith(".flac") or file.lower().endswith(".wav"):
                        speaker_audio.append(
                            os.path.join(root, file)
                        )
            if len(speaker_audio) == 0:
                continue

            self.speaker_to_index[speaker] = speaker_idx
            for path in speaker_audio:
                self.audio_paths.append(path)
                self.labels.append(speaker_idx)
            speaker_idx += 1

        print("=" * 60)
        print("Dataset Loaded")
        print("=" * 60)
        print("Total Speakers :", len(self.speaker_to_index))
        print("Total Audio    :", len(self.audio_paths))
        print("=" * 60)

        if len(self.audio_paths) == 0:
            raise RuntimeError(
                f"No audio files found under:\n{self.root_dir}"
            )
            
    def __len__(self):
        return len(self.audio_paths)

    def _crop_audio(self, waveform):

        length = waveform.size(1)
        if length > self.segment_samples:
            if self.train:
                start = random.randint(
                    0,
                    length - self.segment_samples
                )
            else:

                start = (
                    length - self.segment_samples
                ) // 2
            waveform = waveform[
                :,
                start:start + self.segment_samples
            ]

        elif length < self.segment_samples:
            waveform = F.pad(
                waveform,
                (
                    0,
                    self.segment_samples - length
                )
            )

        return waveform
        
    def __getitem__(self, index):

        audio_path = self.audio_paths[index]
        label = self.labels[index]
        waveform, sr = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True
            )
        if sr != self.sample_rate:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(
                    sr,
                    self.sample_rate
                )
            waveform = self.resamplers[sr](waveform)
        waveform = self._crop_audio(waveform)
        
        if self.train and self.use_augmentation:
            waveform = self.augmentation(waveform)
        features = self.feature_extractor(waveform)
        return features, label
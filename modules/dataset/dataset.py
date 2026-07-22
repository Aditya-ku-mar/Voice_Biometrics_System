import os
import random

import torch
import torchaudio

from torch.utils.data import Dataset

from modules.dataset.features import LogMelFeatureExtractor
from modules.dataset.augmentation import AudioAugmentation


class SpeakerDataset(Dataset):
    """
    Speaker Verification Dataset

    Expected directory structure

    dataset/
    ├── speaker0001/
    │      xxx.wav
    │      yyy.wav
    │
    ├── speaker0002/
    │      aaa.wav
    │      bbb.wav
    │
    └── speaker0003/
           ...

    Returns
    -------
    feature : Tensor
        Shape -> (80, Time)

    label : int
    """

    def __init__(
        self,
        root_dir,
        sample_rate=16000,
        segment_length=3,
        train=True,
        augmentation=True
    ):

        self.root_dir = root_dir

        self.sample_rate = sample_rate

        self.segment_length = segment_length

        self.segment_samples = sample_rate * segment_length

        self.train = train

        self.feature_extractor = LogMelFeatureExtractor(
            sample_rate=sample_rate
        )

        self.augmentation = AudioAugmentation()

        self.use_augmentation = augmentation

        self.audio_paths = []

        self.labels = []

        self.speaker_to_index = {}

        self._load_dataset()

    # ----------------------------------------------------- #
    # Scan Dataset
    # ----------------------------------------------------- #

    def _load_dataset(self):

        speakers = sorted(os.listdir(self.root_dir))

        speaker_index = 0

        for speaker in speakers:

            speaker_path = os.path.join(
                self.root_dir,
                speaker
            )

            if not os.path.isdir(speaker_path):
                continue

            self.speaker_to_index[speaker] = speaker_index

            for file in os.listdir(speaker_path):

                if file.endswith(".wav"):

                    self.audio_paths.append(
                        os.path.join(
                            speaker_path,
                            file
                        )
                    )

                    self.labels.append(
                        speaker_index
                    )

            speaker_index += 1

        print("=" * 60)
        print("Dataset Loaded")
        print("=" * 60)

        print(f"Total Speakers : {len(self.speaker_to_index)}")
        print(f"Total Audio    : {len(self.audio_paths)}")

    # ----------------------------------------------------- #

    def __len__(self):

        return len(self.audio_paths)

    # ----------------------------------------------------- #
    # Crop / Pad Audio
    # ----------------------------------------------------- #

    def _crop_audio(
        self,
        waveform
    ):

        length = waveform.shape[1]

        # Random crop

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
                start:start+self.segment_samples
            ]

        # Pad

        elif length < self.segment_samples:

            shortage = self.segment_samples - length

            waveform = torch.nn.functional.pad(
                waveform,
                (0, shortage)
            )

        return waveform

    # ----------------------------------------------------- #

    def __getitem__(
        self,
        index
    ):

        audio_path = self.audio_paths[index]

        label = self.labels[index]

        waveform, sr = torchaudio.load(audio_path)

        # Convert Stereo → Mono

        if waveform.shape[0] > 1:

            waveform = waveform.mean(
                dim=0,
                keepdim=True
            )

        # Resample

        if sr != self.sample_rate:

            resampler = torchaudio.transforms.Resample(
                sr,
                self.sample_rate
            )

            waveform = resampler(waveform)

        # Crop / Pad

        waveform = self._crop_audio(
            waveform
        )

        # Augmentation

        if self.train and self.use_augmentation:

            waveform = self.augmentation(
                waveform
            )

        # Feature Extraction

        features = self.feature_extractor(
            waveform
        )

        return features, label
import os
import torch
import torchaudio
from torch.utils.data import Dataset

class SpeakerDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.audio_paths = []
        self.labels = []
        self.speaker_to_index = {}
        self._load_dataset()

    def _load_dataset(self):
        speaker_folders = sorted(os.listdir(self.root_dir))
        for speaker_index, speaker_name in enumerate(speaker_folders):
            self.speaker_to_index[speaker_name] = speaker_index
            speaker_path = os.path.join(self.root_dir, speaker_name)
            for file in os.listdir(speaker_path):
                if file.endswith(".wav"):
                    self.audio_paths.append(
                        os.path.join(speaker_path, file)
                    )
                    self.labels.append(speaker_index)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        label = self.labels[index]
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform, label
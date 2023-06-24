import torch
from utils import ACOUSTIC_MEAN, ACOUSTIC_STANDARD_DEVIATION, get_spectrogram, get_true_acoustics

class TVDataset(torch.utils.data.Dataset):

    def __init__(self, audio_paths):
        self.spectrograms = get_spectrogram(audio_paths)
        self.acoustics    = get_true_acoustics(audio_paths, True)
        self.length       = len(self.spectrograms)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x = self.spectrograms[i]
        y = self.acoustics[i]
        return x, y
    
class TDataset(torch.utils.data.Dataset):
    
    def __init__(self, audio_paths):
        self.spectrograms = get_spectrogram(audio_paths)
        self.length = len(self.spectrograms)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x = self.spectrograms[i]
        return x
    
def run(audio_paths, batch_size, shuffle = True, num_workers = 8):

    return torch.utils.data.DataLoader(
                TVDataset(audio_paths), 
                batch_size = batch_size, 
                shuffle = shuffle, 
                num_workers = num_workers,
                pin_memory=True
            )
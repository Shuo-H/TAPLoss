import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from TAPLoss import DEVICE, MODEL_PATH, DEFAULT_N_FFT
import torch

class AcousticEstimator(torch.nn.Module):
    
    def __init__(self):
        super(AcousticEstimator, self).__init__()
        self.lstm = torch.nn.LSTM(DEFAULT_N_FFT + 2 , 256, 4, bidirectional=True, batch_first=True) # DEFAULT_N_FFT = 640
        self.linear1 = torch.nn.Linear(512, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 25)
        self.act = torch.nn.GELU()
        
    def forward(self, spectrogram):
        hidden, _   = self.lstm(spectrogram)
        hidden      = self.linear1(hidden)
        hidden      = self.act(hidden)
        hidden      = self.linear2(hidden)
        hidden      = self.act(hidden)
        acoustics   = self.linear3(hidden)
        return acoustics

def build_infer_model():
    CHECKPOINT = torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict']
    ACOUSTIC_ESTIMATOR = AcousticEstimator()
    ACOUSTIC_ESTIMATOR.load_state_dict(CHECKPOINT)
    return ACOUSTIC_ESTIMATOR.to(DEVICE)

def build_train_model():
    ACOUSTIC_ESTIMATOR = AcousticEstimator()
    return ACOUSTIC_ESTIMATOR.to(DEVICE)
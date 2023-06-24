import torch
import opensmile
LLD = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

### Parameters for Waveform ###
MINIMUM_DURATION_IN_SECONDS = 5 # Waveform Threshold Limits
DEFAULT_SAMPLE_RATE = 16000     # Waveform Default sample rate

### Spectrogram Default Parameters
DEFAULT_N_FFT       = 640
DEFAULT_HOP_LENGTH  = 160
DEFAULT_WIN_LENGTH  = 640
DEFAULT_WINDOW      = torch.hamming_window(DEFAULT_WIN_LENGTH)

### Parameters for LLD features ###
ACOUSTIC_FEATURE_NAMES = LLD.feature_names
ACOUSTIC_MEAN = torch.tensor(
    [  2.31615782e-01, -5.02114248e+00,  7.16793156e+00,  1.40047576e-02,
      -1.44424592e-03,  1.18291244e-01,  7.16937304e+00,  5.01161051e+00,
       7.38044071e+00,  1.30544746e+00,  7.16783571e+00,  7.72617990e-03,
       3.78611624e-01,  1.80594587e+00,  2.74223471e+00,  7.16790104e+00,
       2.29371735e+02,  2.61031281e+02, -2.86713428e+01,  4.58741486e+02,
       2.72984955e+02, -2.86713428e+01,  4.58874390e+02,  2.71175812e+02,
      -2.86713428e+01], dtype=torch.float32)

ACOUSTIC_STANDARD_DEVIATION = torch.tensor(
    [ 4.24716711e-01, 1.09750290e+01, 1.51086359e+01, 2.98775751e-02,
      1.85245797e-02, 2.39421308e-01, 1.63376312e+01, 1.22261524e+01,
      1.53735695e+01, 1.42613926e+01, 1.21981163e+01, 2.58955006e-02,
      8.05543840e-01, 3.83967781e+00, 6.79308844e+00, 1.41308403e+01,
      3.49271667e+02, 6.28384338e+02, 6.05799637e+01, 6.89079407e+02,
      5.62089905e+02, 6.05799637e+01, 1.09140088e+03, 5.42341919e+02,
      6.05799637e+01], dtype=torch.float32)

### Parameters for Model ###
MODEL_PATH = "hamming_lld_estimator_13mse_13mae.pt"
# MODEL_PATH = "/home/shuohan/TAPLoss/hamming_lld_estimator_13mse_13mae.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import gc
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from TAPLoss import *
import torch
import torchaudio
import typing
from tqdm.auto import tqdm
import opensmile
from model import build_infer_model

def get_waveform(
    audioPathList : typing.List[str]
    ) -> torch.FloatTensor:
    
    """
    Parameters:
        audioPathList list(str): 
            List of paths, each to an audio file.

    Returns:
        waveform (torch.FloatTensor): 
            A 2-D time sequence (B x T). 
    """
    waveformList = []
    
    for audioPath in tqdm(audioPathList):
        
        try:
            waveform, sampleRate = torchaudio.load(audioPath)
            
            # Resamples the waveform if necessary at the new frequency using  
            # bandlimited interpolation. [Smith, 2020].
            waveform = torchaudio.functional.resample(
                waveform  = waveform            , 
                orig_freq = sampleRate          , 
                new_freq  = DEFAULT_SAMPLE_RATE )[0]
            
            # Check if waveform duration is less than minimum limit.
            if len(waveform) < MINIMUM_DURATION_IN_SECONDS * DEFAULT_SAMPLE_RATE:
                raise RuntimeWarning(
                    'RuntimeWarning in getSpectrogramFromWaveform: ' +
                    'len(waveform) should be less than {} but received {}'.format(
                        MINIMUM_DURATION_IN_SECONDS * DEFAULT_SAMPLE_RATE, len(waveform)
                    )
                )
                
            waveformList.append(waveform.unsqueeze(0))
            
            del waveform
            gc.collect()
            
        except Exception as e:
            print('Warning in get_waveform:\n {}'.format(e))
            continue
        
    return torch.cat(waveformList, dim=0)
        
        
def get_spectrogram(
    audioPathList : typing.List[str]
    ) -> torch.FloatTensor:
    
    """
    Parameters:
        audioPathList list(str):
            List of paths, each to an audio file.

    Returns:
        spectrogram (torch.FloatTensor): 
            Returns a wrapped complex tensor of size [2*T, 2*(N_FFT+1)], where 
            N_FFT is the number of frequencies where STFT is applied and 
            T is the total number of frames used. 
            Note: Real and imaginary alternate in sequence.
    """
    try:
        waveform = get_waveform(audioPathList)
        # See: https://pytorch.org/docs/stable/generated/torch.stft.html
        spectrogram = torch.stft(
            input          = waveform           , 
            n_fft          = DEFAULT_N_FFT      , 
            hop_length     = DEFAULT_HOP_LENGTH , 
            win_length     = DEFAULT_WIN_LENGTH , 
            window         = DEFAULT_WINDOW     ,
            return_complex = False              )
        
        # Permute to make time first: (B, N_FFT//2+1, T, 2) -> (B, T, N_FFT//2+1, 2)
        spectrogram = spectrogram.permute(0, 2, 1, 3)
        
        # Alternate between corresponding real and imag components over time 
        # [[real, complex], [real, complex], ...] -> [real, complex, real, complex, ...]  
        spectrogram = spectrogram.reshape(spectrogram.size(0), spectrogram.size(1), -1)

        # Remove last 5 frames, whose targets were not available during training
        spectrogram = spectrogram[:, :-5]
        
        return spectrogram

    except Exception as e:
        raise RuntimeError(
            'Exception thrown in get_spectrogramFromWaveform: {}'.format(e))


def get_pred_acoustics(
    audioPathList : typing.List[str] ) -> torch.FloatTensor:
    """
    Parameters:

        estimator (torch.nn.Module): 
            See our ICASSP paper: [Yunyang, et al. 2023].
        
        audioPath (str): 
            List of paths, each to an audio file.

    Returns:

        acoustics (list[torch.FloatTensor]): 
            A list of 25 time series for each audio. Each time series 
            represents an acoustic. Our ICASSP paper: [Yunyang, et al. 2023].
    """


    try:
        estimator   = build_infer_model()
        spectrogram = get_spectrogram(audioPathList)
        estimator.eval()
        with torch.inference_mode():
            acoustics = estimator(spectrogram.to(DEVICE))
        return acoustics
    
    except Exception as e:
        print('Warning in get_pred_acoustics:\n {}'.format(e))

        
def get_true_acoustics(
    audioPathList : typing.List[str] ,
    normalized    : bool
    ) -> torch.FloatTensor:
    
    """
    Parameters:
        audioPathList list(str): 
            List of paths, each to an audio file.

    Returns:
        waveform (torch.FloatTensor): 
            A 2-D time sequence (B x T). 
    """
    acousticsList = []
    
    for audioPath in tqdm(audioPathList):
        
        try:
            waveform, sampleRate = torchaudio.load(audioPath)
            
            # Resamples the waveform if necessary at the new frequency using  
            # bandlimited interpolation. [Smith, 2020].
            waveform = torchaudio.functional.resample(
                waveform  = waveform            , 
                orig_freq = sampleRate          , 
                new_freq  = DEFAULT_SAMPLE_RATE )[0]
            
            # Check if waveform duration is less than minimum limit.
            if len(waveform) < MINIMUM_DURATION_IN_SECONDS * DEFAULT_SAMPLE_RATE:
                raise RuntimeWarning(
                    'RuntimeWarning in getSpectrogramFromWaveform: ' +
                    'len(waveform) should be less than {} but received {}'.format(
                        MINIMUM_DURATION_IN_SECONDS * DEFAULT_SAMPLE_RATE, len(waveform)
                    )
                )
            
            acoustics = LLD.process(signal = waveform, sampling_rate = DEFAULT_SAMPLE_RATE)[2]
            acoustics = torch.from_numpy(acoustics).float()
            acousticsList.append(acoustics.unsqueeze(0))
            
            del acoustics
            gc.collect()
            
        except Exception as e:
            print('Warning in get_true_acoustics:\n {}'.format(e))
            continue
    if normalized:
        return ( torch.cat(acousticsList, dim=0) - ACOUSTIC_MEAN ) / ACOUSTIC_STANDARD_DEVIATION
    else:
        return torch.cat(acousticsList, dim=0)
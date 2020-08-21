# Librosa STFT/Fbank/MFCC in PyTorch
Author: Shimin Zhang

A librosa STFT/Fbank/mfcc feature extration written up in PyTorch using 1D Convolutions.

## Installation

Install easily with pip:`pip install torch_mfcc` or download this repo, `python setup.py install`.

## Usage

```python3
import torch as th
import torch.nn.functional as tf
import scipy
from torch_mfcc import STFT, FBANK, MFCC
import numpy as np
import librosa


def CalMfcc(signal, sr, fft_len, win_hop, win_len,
            n_mfcc=13, center=True, dct_type=2, norm='ortho'):
    spec_amp, n_fft = librosa.spectrum._spectrogram(signal, n_fft=fft_len, hop_length=win_hop, win_length=win_len,
                                                    center=center, power=2.0, window='hann')
    mel_basis = librosa.filters.mel(sr, n_fft)
    S = np.dot(mel_basis, spec_amp)
    fbank = librosa.core.power_to_db(S, top_db=None)
    mfcc_spec = scipy.fftpack.dct(
        fbank, axis=0, type=dct_type, norm=norm)[:n_mfcc]
    return spec_amp, fbank, mfcc_spec


sig = librosa.load(
    librosa.util.example_audio_file(), duration=10.0, offset=30)[0]
device='cpu'
sig_th = th.from_numpy(sig)[None, :].float().to(device)
fft_len = 1024
win_hop = 256
win_len = 1024
window = 'hann'
n_mel = 128
n_mfcc = 13
dct_type = 4
norm = 'ortho'
center = False
sr = 22050

spec_amp, fbank_spec, mfcc_spec = CalMfcc(
    signal=sig,
    sr=sr,
    fft_len=fft_len,
    win_hop=win_hop,
    win_len=win_len,
    n_mfcc=n_mfcc,
    center=center,
    dct_type=dct_type,
    norm=norm)

# librosa STFT VS conv_stft
# see more test at :https://github.com/echocatzh/conv-stft/blob/master/tests/test_stft.py
stft = STFT(
    win_len=win_len,
    win_hop=win_hop,
    fft_len=fft_len,
    pad_center=center,
    win_type=window).to(device)
real, imag = stft.transform(sig_th, return_type='realimag')
spectrum_th = th.square(real) + th.square(imag)
print(tf.mse_loss(th.FloatTensor(spec_amp).to(device),
                  spectrum_th.squeeze(0)))  # 8.4275e-10

# librosa fbank VS torch_fbank
fbank = FBANK(win_len, win_hop, fft_len, sr,  win_type=window,
              top_db=None,
              center=center, n_mel=n_mel).to(device)
fbank_spec_th = fbank(sig_th)
print(tf.mse_loss(th.FloatTensor(fbank_spec).to(device),
                  fbank_spec_th.transpose(1, 2).squeeze(0)))  # 1.4462e-09

# librosa mfcc VS torch_mfcc
mfcc = MFCC(win_len, win_hop, fft_len, sr, win_type=window,
            top_db=None, dct_type=dct_type, norm=norm,
            center=center, n_mfcc=n_mfcc, n_mel=n_mel).to(device)
mfcc_spec_th = mfcc(sig_th)
print(tf.mse_loss(th.FloatTensor(mfcc_spec).to(device),
                  mfcc_spec_th.transpose(1, 2).squeeze(0)))  # 7.2581e-10

```

## Contact
If you have any questions, welcome to contact me at shmzhang@npu-aslp.org
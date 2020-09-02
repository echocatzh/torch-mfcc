import torch as th
import torch.nn.functional as tf
import numpy as np
from .conv_stft import STFT
from librosa.filters import mel as filters_mel


class FBANK(th.nn.Module):
    def __init__(self, win_len, win_hop, fft_len, sr,
                 win_type='hann', power=2.0, n_mel=128,
                 ref=1.0, amin=1e-17, top_db=80.0, center=True):
        """
        Implement of Fbank using 1D convolution and 1D transpose convolutions.

        Args:
            win_len (int): See `conv_stft.STFT` for details.
            win_hop (int): As above
            fft_len (int): As above
            sr (int): Sample rate of signal
            win_type (str, optional): see `conv_stft.STFT` for details.
            Defaults to 'hann'.
            power (float, optional): Do power to spectrum. Defaults to 2.0.
            n_mel (int, optional): Number of mel filter banks. Defaults to 128.
            ref (float, optional): See `librosa.core.power_to_db` for details.
            amin (float, optional): As above. Defaults to 1e-17.
            top_db (float, optional): As above. Defaults to 80.0.
            center (bool, optional): If padding the input signal. Defaults to True.
        """
        super(FBANK, self).__init__()
        self.stft = STFT(
            win_len=win_len,
            win_hop=win_hop,
            fft_len=fft_len,
            pad_center=center,
            win_type=win_type,
            enframe_mode='break')
        self.sr = sr
        self.fft_len = fft_len
        self.power = power
        self.n_mel = n_mel
        mel_k = self.__init_kernel__()
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        self.register_buffer('mel_k', mel_k)

    def power2db(self, spec):
        """
        See `librosa.core.power_to_db`. Function as dB-scale.

        Args:
            spec (tensors): Input fbank Feature.

        Returns:
            tensors: db-scale of input feature.
        """
        assert self.amin > 0
        amin = th.ones_like(spec)*self.amin
        ref_value = th.ones_like(spec)*np.abs(self.ref)
        log_spec = 10.0*th.log10(th.max(amin, spec))
        log_spec -= 10.0*th.log10(th.max(amin, ref_value))
        if self.top_db is not None:
            assert self.top_db > 0
            peak_th = th.ones_like(log_spec)*(th.max(log_spec)-self.top_db)
            log_spec = th.max(log_spec, peak_th)
        return log_spec

    def __init_kernel__(self):
        """
        Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

        Returns:
            tensors: A Filterbank matrix
        """
        mel_kernel = th.FloatTensor(filters_mel(
            self.sr, self.fft_len, n_mels=self.n_mel))
        return mel_kernel

    def forward(self, inputs):
        """
        Take input data (audio) to fbank feature.

        Args:
            inputs (tensor): Tensor of floats, with shape [num_batch, num_samples]

        Returns:
            tensor: fbank feature of shape [num_batch, num_mels]
        """
        real, imag = self.stft.transform(inputs, return_type='realimag')
        amp_feature = th.sqrt((real)**2 + (imag)**2)
        amp_feature_power = th.pow(amp_feature, self.power)
        mel_spec = tf.linear(amp_feature_power.transpose(1, 2), self.mel_k)
        mel_spec = self.power2db(mel_spec)
        return mel_spec

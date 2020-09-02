import torch as th
import numpy as np
from .torch_fbank import FBANK
import torch.nn.functional as tf


class MFCC(th.nn.Module):
    def __init__(self, win_len, win_hop, fft_len, sr,
                 win_type='hann', power=2.0, n_mel=128, n_mfcc=13,
                 dct_type=2, norm='ortho', ref=1.0,
                 amin=1e-10, top_db=80.0,
                 center=True, enframe_mode='continue'):
        """
        Implement of MFCC using 1D convolution and 1D transpose convolutions.

        Args:
            win_len (int): See `conv_stft.STFT` for details.
            win_hop (int): As above
            fft_len (int): As above
            sr (int): Sample rate of signal
            win_type (str, optional): see `conv_stft.STFT` for details.
            Defaults to 'hann'.
            power (float, optional): Do power to spectrum. Defaults to 2.0.
            n_mel (int, optional): Number of mel filter banks. Defaults to 128.
            n_mfcc (int, optional): Mel Cepstral Coefficient. Defaults to 13.
            dct_type (int, optional): Type of DCT transform. Defaults to 2.
            norm (str, optional): Do normalization. Defaults to 'ortho'.
            ref (float, optional): See `librosa.core.power_to_db` for details.
            amin (float, optional): As above. Defaults to 1e-17.
            top_db (float, optional): As above. Defaults to 80.0.
            center (bool, optional): If padding the input signal.
            Defaults to True.
            enframe_mode (str, optional): default enframe method in librosa.
        """
        super(MFCC, self).__init__()
        self.fbank = FBANK(
            win_len, win_hop, fft_len, sr,
            win_type, power, n_mel, ref, amin, top_db,
            center=center, enframe_mode=enframe_mode)
        self.n_mel = n_mel
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        dct_k = self.__init_kernel__()
        self.register_buffer('dct_k', dct_k.T)

    def __init_kernel__(self):
        """
        Return the Discrete Cosine Transform of arbitrary type sequence x.
        The details of four kind of DCT can be viewed here:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
        Type 1 is not supported, for memory considerations.

        Returns:
            tensors: The DCt transform matrix
        """
        n = th.arange(self.n_mel)[:, None].float()
        k = th.arange(self.n_mfcc)[:, None].float()
        dct_kernel = None
        if self.dct_type == 2:
            dct_kernel = 2.0 * \
                th.cos(np.pi*th.matmul(2.*n+1.0, k.T)/2./self.n_mel)
            if self.norm == 'ortho':
                dct_kernel[:, 0] *= np.sqrt(1/self.n_mel/4.)
                dct_kernel[:, 1:] *= np.sqrt(1/self.n_mel/2.)
        elif self.dct_type == 3:
            dct_kernel = 2.0*th.cos(np.pi*th.matmul(n, 2*k.T+1)/(2*self.n_mel))
            dct_kernel[0, :] = 0.
            if self.norm == 'ortho':
                dct_kernel *= np.sqrt(1/(2*self.n_mel))
        elif self.dct_type == 4:
            dct_kernel = 2.0 * \
                th.cos(np.pi*th.matmul(2.*n+1.0, 2.*k.T+1)/4./self.n_mel)
            if self.norm == 'ortho':
                dct_kernel *= np.sqrt(1/self.n_mel/2)
        else:
            raise RuntimeError(
                "Type {} is not supported.".format(self.dct_type))
        return dct_kernel

    def forward(self, inputs):
        """
        Take input data (audio) to mfcc feature.

        Args:
            inputs (tensor): Tensor of floats, with shape
            [num_batch, num_samples]

        Returns:
            tensor: mfcc feature of shape [num_batch, num_mfcc]
        """
        fbank_spec = self.fbank(inputs)
        if self.dct_type == 3:  # dct_type == 3
            mfcc_spec = tf.linear(fbank_spec, self.dct_k)
            if self.norm != 'ortho':
                mfcc_spec += fbank_spec[:, :, 0:1]
            else:
                mfcc_spec += fbank_spec[:, :, 0:1]*np.sqrt(1/self.n_mel)
        else:  # dct_type == 2/4
            mfcc_spec = tf.linear(fbank_spec, self.dct_k)
        return mfcc_spec

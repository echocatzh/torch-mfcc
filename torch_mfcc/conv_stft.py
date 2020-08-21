import torch
import torch.nn.functional as tf
from scipy.signal import check_COLA, get_window


class STFT(torch.nn.Module):
    def __init__(self, win_len=1024, win_hop=512, fft_len=1024,
                 enframe_mode='continue', win_type='hann',
                 win_sqrt=False, pad_center=True):
        """
        Implement of STFT using 1D convolution and 1D transpose convolutions.
        Implement of framing the signal in 2 ways, `break` and `continue`.
        `break` method is a kaldi-like framing.
        `continue` method is a librosa-like framing.

        Args:
            win_len (int): Number of points in one frame.  Defaults to 1024.
            win_hop (int): Number of framing stride. Defaults to 512.
            fft_len (int): Number of DFT points. Defaults to 1024.
            enframe_mode (str, optional): `break` and `continue`.
            Defaults to 'continue'.
            win_type (str, optional): The type of window to create.
            Defaults to 'hann'.
            win_sqrt (bool, optional): using square root window.
            Defaults to True.
            pad_center (bool, optional): `perfect reconstruction` opts.
            Defaults to True.
        """
        super(STFT, self).__init__()
        assert enframe_mode in ['break', 'continue']
        assert fft_len >= win_len
        self.win_len = win_len
        self.win_hop = win_hop
        self.fft_len = fft_len
        self.mode = enframe_mode
        self.win_type = win_type
        self.win_sqrt = win_sqrt
        self.pad_center = pad_center
        self.pad_amount = self.fft_len // 2

        en_k, fft_k, ifft_k, ola_k = self.__init_kernel__()
        self.register_buffer('en_k', en_k)
        self.register_buffer('fft_k', fft_k)
        self.register_buffer('ifft_k', ifft_k)
        self.register_buffer('ola_k', ola_k)

    def __init_kernel__(self):
        """
        Generate enframe_kernel, fft_kernel, ifft_kernel
        and overlap-add kernel.
        ** enframe_kernel: Using conv1d layer and identity matrix.
        ** fft_kernel: Using linear layer for matrix multiplication. In fact,
        enframe_kernel and fft_kernel can be combined, But for the sake of
        readability, I took the two apart.
        ** ifft_kernel, pinv of fft_kernel.
        ** overlap-add kernel, just like enframe_kernel, but transposed.

        Returns:
            tuple: four kernels.
        """
        enframed_kernel = torch.eye(self.fft_len)[:, None, :]
        fft_kernel = torch.rfft(torch.eye(self.fft_len), 1)
        if self.mode == 'break':
            enframed_kernel = torch.eye(self.win_len)[:, None, :]
            fft_kernel = fft_kernel[:self.win_len]
        fft_kernel = torch.cat(
            (fft_kernel[:, :, 0], fft_kernel[:, :, 1]), dim=1)
        ifft_kernel = torch.pinverse(fft_kernel)[:, None, :]
        window = get_window(self.win_type, self.win_len)

        self.perfect_reconstruct = check_COLA(
            window,
            self.win_len,
            self.win_len-self.win_hop)
        window = torch.FloatTensor(window)
        if self.mode == 'continue':
            left_pad = (self.fft_len - self.win_len)//2
            right_pad = left_pad + (self.fft_len - self.win_len) % 2
            window = tf.pad(window, (left_pad, right_pad))
        if self.win_sqrt:
            self.padded_window = window
            window = torch.sqrt(window)
        else:
            self.padded_window = window**2

        fft_kernel = fft_kernel.T * window
        ifft_kernel = ifft_kernel * window
        ola_kernel = torch.eye(self.fft_len)[:self.win_len, None, :]
        if self.mode == 'continue':
            ola_kernel = torch.eye(self.fft_len)[:, None, :self.fft_len]
        return enframed_kernel, fft_kernel, ifft_kernel, ola_kernel

    def is_perfect(self):
        """
        Whether the parameters win_len, win_hop and win_sqrt
        obey constants overlap-add(COLA)

        Returns:
            bool: Return true if parameters obey COLA.
        """
        return self.perfect_reconstruct and self.pad_center

    def transform(self, inputs, return_type='magphase'):
        """Take input data (audio) to STFT domain.

        Args:
            inputs (tensor): Tensor of floats,
            with shape (num_batch, num_samples)
            return_type (str, optional): return (mag, phase) when `magphase`,
            return (real, imag) when `realimag`. Defaults to 'magphase'.

        Returns:
            tuple: (mag, phase) when `magphase`, return (real, imag) when
            `realimag`. Defaults to 'magphase', each elements with shape
            [num_batch, num_frequencies, num_frames]
        """
        assert return_type in ['magphase', 'realimag']
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        self.num_samples = inputs.size(-1)
        if self.pad_center:
            inputs = tf.pad(
                inputs, (self.pad_amount, self.pad_amount), mode='reflect')
        enframe_inputs = tf.conv1d(inputs, self.en_k, stride=self.win_hop)
        outputs = torch.transpose(enframe_inputs, 1, 2)
        outputs = tf.linear(outputs, self.fft_k)
        outputs = torch.transpose(outputs, 1, 2)
        dim = self.fft_len//2+1
        real = outputs[:, :dim, :]
        imag = outputs[:, dim:, :]
        if return_type == 'realimag':
            return real, imag
        else:
            mags = torch.sqrt(real**2+imag**2)
            phase = torch.atan2(imag, real)
            return mags, phase

    def inverse(self, input1, input2, input_type='magphase'):
        """Call the inverse STFT (iSTFT), given tensors produced
        by the `transform` function.

        Args:
            input1 (tensors): Magnitude/Real-part of STFT with shape
            [num_batch, num_frequencies, num_frames]
            input2 (tensors): Phase/Imag-part of STFT with shape [
            [num_batch, num_frequencies, num_frames]
            input_type (str, optional): Mathematical meaning of input tensor's.
            Defaults to 'magphase'.

        Returns:
            tensors: Reconstructed audio given magnitude and phase. Of
                shape [num_batch, num_samples]
        """
        assert input_type in ['magphase', 'realimag']
        if input_type == 'realimag':
            real, imag = input1, input2
        else:
            real = input1*torch.cos(input2)
            imag = input1*torch.sin(input2)
        inputs = torch.cat([real, imag], dim=1)
        outputs = tf.conv_transpose1d(inputs, self.ifft_k, stride=self.win_hop)
        t = (self.padded_window[None, :, None]).repeat(1, 1, inputs.size(-1))
        t = t.to(inputs.device)
        coff = tf.conv_transpose1d(t, self.ola_k, stride=self.win_hop)
        rm_start, rm_end = self.pad_amount, self.pad_amount+self.num_samples
        outputs = outputs[..., rm_start:rm_end]
        coff = coff[..., rm_start:rm_end]
        coffidx = torch.where(coff > 1e-8)
        outputs[coffidx] = outputs[coffidx]/(coff[coffidx])
        return outputs.squeeze(dim=1)

    def forward(self, inputs):
        """Take input data (audio) to STFT domain and then back to audio.

        Args:
            inputs (tensor): Tensor of floats,
            with shape [num_batch, num_samples]

        Returns:
            tensor: Reconstructed audio given magnitude and phase.
            Of shape [num_batch, num_samples]
        """
        mag, phase = self.transform(inputs)
        rec_wav = self.inverse(mag, phase)
        return rec_wav

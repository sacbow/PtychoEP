from __future__ import annotations
from .uncertain_array import UncertainArray as UA
from .uncertain_array import fft_ua, ifft_ua
from ...backend import np
from ...ptycho.data import DiffractionData
from .object import Object


class FFTChannel:
    """
    Object → FFT → Data、またはその逆の伝搬を担当するチャネル。

    Attributes:
        msg_from_input (UA): FFT入力（Object → Data）のメッセージを保持。
        cached_output (UA): backward結果をキャッシュ（必要に応じて）
    """

    def __init__(self, rng):
        self.msg_from_input: UA = None # UA with array precision
        self.msg_from_output: UA = None # UA with scalar precision
        self.output_belief: UA = None # UA with scalar precision
        self.rng = rng
    
    def receive_msg_from_input(self, incoming : UA):
        self.msg_from_input = incoming
    
    def receive_msg_from_output(self, incoming : UA):
        self.msg_from_output = incoming

    def forward(self) -> UA:
        """
        ObjectからのメッセージをFFTしてData側に送る。
        """
        if self.msg_from_input is None:
            raise RuntimeError("no input message to forward")
        if self.output_belief is None or self.msg_from_output is None:
            return UA.normal(rng = self.rng, shape = self.msg_from_input.shape, scalar_precision=True)
        else:
            return self.output_belief/self.msg_from_output

    def backward(self) -> UA:
        """
        IFFTを通じて、DataからObjectへのメッセージを復元。
        """
        if self.msg_from_output is None or self.msg_from_input is None:
            raise RuntimeError("No message to backward.")
        # --compute internal belief--
        msg_to_backward = ifft_ua(self.msg_from_output)
        input_belief = self.msg_from_input * msg_to_backward.to_array_precision()
        self.output_belief = fft_ua(input_belief)
        # backward
        return msg_to_backward

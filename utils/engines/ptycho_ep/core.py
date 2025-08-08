# utils/engines/ptycho_ep/core.py

from ...backend import np
from ...rng_utils import get_rng, normal
from ...ptycho.data import DiffractionData
from .object import Object
from .probe import Probe
from .fft_channel import FFTChannel
from .denoiser import PROutputDenoiser, GaussianPriorDenoiser, SparsePriorDenoiser

class PtychoEP:
    def __init__(self, ptycho, prior_type="gaussian", prior_kwargs=None,
                 damping=1.0, prb_init=None, callback=None):
        """
        EP-based ptychography solver.

        Parameters
        ----------
        ptycho : Ptycho
            Forward済みのPtychoインスタンス。
        prior_type : str
            "gaussian" または "sparse"。
            "gaussian" の場合、AUA の初期精度=1.0 が平均0・精度1のガウス事前に
            相当するため、prior ノードを構築せず prior 更新は行わない。
            （差し替え処理を行っても状態は変わらないため、省略することで
             不要な演算と数値誤差を避ける。）
            "sparse" の場合は SparsePriorDenoiser を用いて G1 ノード更新を行う。
        prior_kwargs : dict
            PriorDenoiser に渡す追加パラメータ（例: {"rho": 0.1}）。
        damping : float
            メッセージ更新の減衰係数。
        prb_init : np.ndarray or None
            プローブ初期値。None なら ptycho.prb をコピー。
        callback : callable
            各反復終了時に呼び出される関数 (iter, err, obj_est)。
        """

        self.xp = np()
        self.ptycho = ptycho
        self.damping = damping
        self.callback = callback

        xp = self.xp
        rng = get_rng()

        # --- object init ---
        self.obj_node = Object(shape=(ptycho.obj_len, ptycho.obj_len), rng=rng)

        # --- probe init ---
        self.probe = Probe(prb_init if prb_init is not None else ptycho.prb)

        # --- channels & output nodes ---
        self.fft_channels = {}
        self.output_nodes = {}
        for d in ptycho._diff_data:
            self.obj_node.register_data(d)
            ch = FFTChannel(rng=rng)
            self.fft_channels[d] = ch
            gamma_w = d.gamma_w if d.gamma_w is not None else 1.0 
            out = PROutputDenoiser(shape=d.diffraction.shape, data=d,
                                  gamma_w=gamma_w, damping=damping)
            self.output_nodes[d] = out

        # --- prior node ---
        prior_kwargs = prior_kwargs or {}
        if prior_type == "gaussian":
            self.prior = None
        elif prior_type == "sparse":
            self.prior = SparsePriorDenoiser(shape=(ptycho.obj_len, ptycho.obj_len),
                                             damping=damping, **prior_kwargs)
        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")

    def run(self, n_iter=100):
        """
        逐次EPループ
        """
        xp = self.xp
        for it in range(n_iter):
            # --- 各データ点ごとの G2->G3 更新 ---
            for d in self.ptycho._diff_data:
                # Object -> Probe
                msg_obj_to_data = self.obj_node.send_msg_to_data(d)
                msg_after_probe = self.probe.forward(msg_obj_to_data)

                # FFT forward
                ch = self.fft_channels[d]
                ch.receive_msg_from_input(msg_after_probe)
                msg_to_output = ch.forward()

                # Output denoiser
                out = self.output_nodes[d]
                out.receive_msg(msg_to_output)
                msg_from_output = out.forward_msg()

                # FFT backward
                ch.receive_msg_from_output(msg_from_output)
                msg_back_fft = ch.backward()

                # Probe backward
                msg_back_probe = self.probe.backward(msg_back_fft)

                # Object受信
                self.obj_node.receive_msg_from_data(d, msg_back_probe)

            # --- G1更新（Prior） ---
            if self.prior is not None:
                self.prior.receive_msg(self.obj_node.send_msg_to_prior())
                msg_prior = self.prior.forward_msg()
                self.obj_node.receive_msg_from_prior(msg_prior)

            # --- callback ---
            if self.callback:
                mean_err = xp.mean([out.error for out in self.output_nodes.values()])
                self.callback(it, float(mean_err), self.obj_node.get_belief().mean)

        return self.obj_node.get_belief().mean, self.probe.data

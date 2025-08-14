from .accumulative_uncertain_array import AccumulativeUncertainArray as AUA
from .uncertain_array import UncertainArray as UA
from ...backend import np
from ...ptycho.data import DiffractionData  # 実際のインポートパスに応じて調整


class Object:
    def __init__(self, shape, rng, dtype=np().complex64, init_ua: UA | None = None):
        self.shape = shape
        self.dtype = dtype
        self.belief = AUA(shape=shape, dtype=dtype)
        self.rng = rng

        # ★ 全域の初期UA（未指定ならランダムで生成）
        #   空間的に相関のある“共通の場”から、各パッチの初期メッセージを切り出す
        self.init_ua: UA = init_ua if init_ua is not None \
            else UA.normal(shape=shape, rng=self.rng, scalar_precision=False)

        self.msg_from_prior: UA = UA.zeros(shape=shape, scalar_precision=False)
        self.msg_from_data: dict[DiffractionData, UA] = {}
        self.data_registry: dict[DiffractionData, tuple[slice, slice]] = {}

    def register_data(self, data: DiffractionData, probe: np().ndarray | None = None,
                      precision_floor: float = 0.0):
        if data.indices is None:
            raise ValueError(f"indices not set for data at position {data.position}")
        self.data_registry[data] = data.indices

        # ★ 全域 init_ua から該当パッチを切り出す
        ua0 = self.init_ua[data.indices]   # UA を返す（__getitem__）

        # （任意）プローブで重み付けしたい場合：scaled を併用
        if probe is not None:
            # mean × probe, precision × |probe|^2（ゼロ強度は floor に）
            ua0 = ua0.scaled(probe, precision_floor=precision_floor)

        self.msg_from_data[data] = ua0
        self.belief.add(ua0, data.indices)

    def receive_msg_from_data(self, data: DiffractionData, msg: UA):
        indices = self.data_registry.get(data)
        if indices is None:
            raise ValueError("Data not registered to object")

        prev_msg = self.msg_from_data.get(data)
        if prev_msg is not None:
            self.belief.subtract(prev_msg, indices)
        self.belief.add(msg, indices)
        self.msg_from_data[data] = msg

    def receive_msg_from_prior(self, msg: UA):
        self.belief.subtract(self.msg_from_prior)
        self.belief.add(msg)
        self.msg_from_prior = msg

    def get_patch_ua(self, data: DiffractionData) -> UA:
        indices = self.data_registry.get(data)
        if indices is None:
            raise ValueError("Data not registered to object")
        return self.belief.get_ua(indices)
    
    def send_msg_to_data(self, data: DiffractionData) -> UA:
        belief_patch = self.get_patch_ua(data)
        incoming_msg = self.msg_from_data[data]
        msg_to_send = belief_patch/incoming_msg
        return msg_to_send

    def send_msg_to_prior(self) -> UA:
        if self.msg_from_prior is None:
            raise RuntimeError("Prior message not set.")
        return self.belief.to_ua() / self.msg_from_prior

    def get_belief(self) -> UA:
        return self.belief.to_ua()

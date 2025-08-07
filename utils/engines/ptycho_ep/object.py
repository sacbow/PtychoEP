from .accumulative_uncertain_array import AccumulativeUncertainArray as AUA
from .uncertain_array import UncertainArray as UA
from ...backend import np
from ...ptycho.data import DiffractionData  # 実際のインポートパスに応じて調整


class Object:
    def __init__(self, shape, rng, dtype=np().complex64):
        self.shape = shape
        self.dtype = dtype
        self.belief = AUA(shape=shape, dtype=dtype)
        self.rng = rng
        self.msg_from_prior: UA = UA.zeros(shape = shape)
        self.msg_from_data: dict[DiffractionData, UA] = {}
        self.data_registry: dict[DiffractionData, tuple[slice, slice]] = {}

    def register_data(self, data: DiffractionData):
        """
        DiffractionData オブジェクトを登録し、スライス位置情報を保持。
        """
        if data.indices is None:
            raise ValueError(f"indices not set for data at position {data.position}")
        self.data_registry[data] = data.indices
        random_msg = UA.normal(rng = self.rng, shape = data.diffraction.shape, scalar_precision= False)
        self.msg_from_data[data] = random_msg
        self.belief.add(random_msg, data.indices)

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
        self.msg_from_prior = msg

    def get_patch_ua(self, data: DiffractionData) -> UA:
        indices = self.data_registry.get(data)
        if indices is None:
            raise ValueError("Data not registered to object")
        return self.belief.get_ua(indices)
    
    def send_msg_to_data(self, data: DiffractionData) -> UA:
        belief_patch = self.get_patch_ua(self, data)
        incoming_msg = self.msg_from_data[data]
        msg_to_send = belief_patch/incoming_msg
        return msg_to_send

    def send_msg_to_prior(self) -> UA:
        if self.msg_from_prior is None:
            raise RuntimeError("Prior message not set.")
        return self.belief.to_ua() / self.msg_from_prior

    def get_belief(self) -> UA:
        return self.belief.to_ua()

import torch
import numpy as np
from config import Config


class Tensor():
    def __init__(self, tensor):
        self.tensor = tensor

    def torch_tensor_for_device(self):
        tensor = self.tensor
        if isinstance(tensor, torch.Tensor):
            return tensor
        tensor = np.asarray(tensor, dtype=np.float32)
        tensor = torch.from_numpy(tensor).to(Config.DEVICE)
        return tensor

    def to_np(self):
        return self.tensor.cpu().detach().numpy()

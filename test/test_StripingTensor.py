
import torch

from StripingTensor import StripingTensor

def test_stripingTensor():
    srpTensor = StripingTensor(torch.empty(55, 4, 5, dtype=torch.float32), 10, dtype=torch.float32, device=torch.device('cpu'))

    print(srpTensor.stripe_border)
    print(srpTensor.mutex)

test_stripingTensor()
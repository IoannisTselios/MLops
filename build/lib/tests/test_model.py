import torch
from dog_breed_identification.models.model import Model


def test_model():
    test_input = torch.randn(1, 3, 200, 200)
    model = Model()
    out = model(test_input)
    assert out.shape == torch.Size([1, 120]), "Output shape is incorrect"

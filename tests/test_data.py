import os

import pytest
import torch
from tests import _PATH_DATA, _PROJECT_ROOT
import hydra
from torch.utils.data import DataLoader
from dog_breed_identification.data.load_data import LoadData
from hydra.core.global_hydra import GlobalHydra
config_path = os.path.join(_PROJECT_ROOT, 'config')

if not GlobalHydra().is_initialized():
    hydra.initialize(config_path="../dog_breed_identification/config", version_base=None)
train_config = hydra.compose(config_name="train_config")
dataset_path = train_config.data_path
batch_size = train_config.batch_size

if os.path.exists(f"{dataset_path}/train.pt"):
    train = torch.load("data/processed/train.pt")
    val = torch.load("data/processed/val.pt")
else:
    train = LoadData.load(f'{dataset_path}/train.pt', train_config)
    val = LoadData.load(f'{dataset_path}/val.pt', train_config)
# Create dataloaders
train_ldr = DataLoader(train, batch_size=batch_size, shuffle=True)
test_ldr = DataLoader(val, batch_size=batch_size, shuffle=True)


# @pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    assert len(train_ldr) > 0, "Dataset did not have the correct number of samples"
    assert len(test_ldr) > 0, "Dataset did not have the correct number of samples"
    assert len(train_ldr) > len(
        test_ldr), "Train set should be larger than test set"
    assert train_ldr.dataset[0][0].shape == torch.Size(
        [3, 200, 200]), "Input data shape is incorrect"

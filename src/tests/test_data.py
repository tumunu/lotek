# test_data.py - Pytest for data

import pytest
from src.data import get_data_loader, collate_fn
from datasets import Dataset

@pytest.fixture
def mock_dataset():
    return Dataset.from_dict({"text": ["This is a test", "Another test"]})

def test_get_data_loader(mock_dataset):
    loader = get_data_loader(2)
    assert isinstance(loader, DataLoader)
    for batch in loader:
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        break  # Just check the first batch

def test_collate_fn():
    # Mock data
    batch = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5], "attention_mask": [1, 1]}
    ]
    result = collate_fn(batch)
    assert len(result['input_ids']) == 2
    assert len(result['attention_mask']) == 2
    assert result['input_ids'].shape[1] == 3  # Padded to the longest sequence
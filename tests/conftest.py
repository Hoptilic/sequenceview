import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sequenceview.model import ProteinClassifier  # noqa: E402

LYSOZYME = (
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
)


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    model = ProteinClassifier(vocab_size=22, embedding_dim=8, hidden_dim=8)
    path = tmp_path_factory.mktemp("model") / "tiny.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "max_length": 64,
            "vocab_size": 22,
            "metadata": {"oxidoreductase_count": 10, "hydrolase_count": 10},
            "temperature": 1.5,
        },
        path,
    )
    return path


@pytest.fixture()
def client(checkpoint, monkeypatch):
    monkeypatch.setenv("SEQUENCEVIEW_CHECKPOINT_PATH", str(checkpoint))
    from sequenceview.app import create_app

    return create_app().test_client()


@pytest.fixture()
def lysozyme():
    return LYSOZYME

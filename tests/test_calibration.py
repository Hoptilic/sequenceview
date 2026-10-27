import torch

from sequenceview.calibration import expected_calibration_error, fit_temperature


def test_temperature_above_one_for_overconfident_logits():
    torch.manual_seed(1)
    labels = torch.randint(0, 2, (2000,))
    signal = torch.where(labels == 1, 1.0, -1.0)
    logits = torch.stack([-signal, signal], dim=1) * 1.0 + torch.randn(2000, 2)
    overconfident = logits * 4

    temperature = fit_temperature(overconfident, labels)
    assert temperature > 1.5

    before = expected_calibration_error(torch.softmax(overconfident, dim=1), labels)
    after = expected_calibration_error(torch.softmax(overconfident / temperature, dim=1), labels)
    assert after < before


def test_ece_is_zero_for_perfect_confident_predictions():
    labels = torch.tensor([0, 1, 1, 0])
    probabilities = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    assert expected_calibration_error(probabilities, labels) == 0.0

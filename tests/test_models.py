from __future__ import annotations

import pytest
import torch

from robotics_vision.models import build_model, count_parameters


@pytest.mark.parametrize("architecture", ["tiny_unet", "lite_fcn"])
def test_models_preserve_spatial_shape_and_script(architecture: str) -> None:
    model = build_model({"architecture": architecture, "num_classes": 4, "base_channels": 4})
    model.eval()
    example = torch.rand(2, 3, 31, 35)
    with torch.inference_mode():
        output = model(example)
        scripted_output = torch.jit.script(model)(example)
    assert output.shape == (2, 4, 31, 35)
    assert torch.allclose(output, scripted_output)
    assert count_parameters(model) > 0

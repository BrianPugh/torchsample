import torch
import torch.nn.functional as F
import torchtest

import torchsample as ts


def test_mlp():
    batch = [
        torch.rand(7, 4096, 10),
        torch.rand(7, 4096, 3),
    ]
    model = ts.models.MLP([10, 100, 100, 3])
    torchtest.assert_vars_change(
        model=model,
        loss_fn=F.l1_loss,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        device="cpu",
    )
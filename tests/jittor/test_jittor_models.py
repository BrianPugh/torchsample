import jittor as jt
from jittor import nn

import torchsample.jittor as ts


def allclose(actual, expected):
    return ((expected - actual).abs() < 1e-5).all()


def test_mlp():
    x_gt = jt.rand(7, 4096, 10)
    y_gt = jt.rand(7, 4096, 3)

    model = ts.models.MLP(10, 100, 100, 3)
    og_parameters = [x.clone() for x in model.parameters()]
    optim = nn.SGD(model.parameters(), 1e-2)

    pred_y = model(x_gt)
    dy = pred_y - y_gt
    loss = dy * dy
    loss_mean = loss.mean()
    optim.step(loss_mean)

    for og, optimized in zip(og_parameters, model.parameters()):
        assert not allclose(og, optimized)


def test_mlp_list():
    x_gt = jt.rand(7, 4096, 10)
    y_gt = jt.rand(7, 4096, 3)

    model = ts.models.MLP([10, 100, 100, 3])
    og_parameters = [x.clone() for x in model.parameters()]
    optim = nn.SGD(model.parameters(), 1e-2)

    pred_y = model(x_gt)
    dy = pred_y - y_gt
    loss = dy * dy
    loss_mean = loss.mean()
    optim.step(loss_mean)

    for og, optimized in zip(og_parameters, model.parameters()):
        assert not allclose(og, optimized)

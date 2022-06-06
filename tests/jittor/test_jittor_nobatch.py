import jittor as jt
import pytest

from torchsample.jittor._nobatch import nobatch


def assert_close(actual, expected):
    assert ((expected - actual).abs() < 1e-4).all()


@pytest.fixture
def singleton_obj():
    return object()


def test_nobatch_nontensor(singleton_obj):
    @nobatch
    def foo(x):
        return x

    assert foo(singleton_obj) == singleton_obj
    assert foo.nobatch(singleton_obj) == singleton_obj


def test_nobatch_tensor_onedim_args():
    @nobatch
    def foo(x):
        assert x.shape == (1, 3)
        return x

    data = jt.ones(3)
    assert_close(foo.nobatch(data), data)


def test_nobatch_tensor_onedim_kwargs():
    @nobatch
    def foo(x=None):
        assert x.shape == (1, 3)
        return x

    data = jt.ones(3)
    assert_close(foo.nobatch(x=data), data)


def test_nobatch_tensor_return_tuple():
    @nobatch
    def foo(x, y):
        assert x.shape == (1, 3)
        assert y.shape == (1, 4, 2)
        return x, y

    data1 = jt.ones(3)
    data2 = jt.zeros((4, 2))
    out1, out2 = foo.nobatch(data1, data2)

    assert_close(data1, out1)
    assert_close(data2, out2)

import pytest

from pttp import TensorProfiler


def test_instance():
    instance = TensorProfiler()
    assert TensorProfiler.instance() is instance


def test_instances():
    assert TensorProfiler.instances() == []
    instance_0 = TensorProfiler()
    instance_1 = TensorProfiler()
    assert TensorProfiler.instances() == [instance_0, instance_1]


def test_instance_context():
    with TensorProfiler() as prof:
        assert TensorProfiler.instance() is prof
    assert TensorProfiler.instance() is prof


def test_instance_errors():
    with pytest.raises(ValueError):
        TensorProfiler.instance()

    instance_0 = TensorProfiler()
    instance_1 = TensorProfiler()

    with pytest.raises(ValueError):
        TensorProfiler.instance()


def test_instances_collection():
    instance = TensorProfiler()
    assert TensorProfiler.instances() == [instance]

    del instance
    assert TensorProfiler.instances() == []

from pttp import TensorProfiler

def test_instance():
    instance = TensorProfiler()
    assert TensorProfiler.instance() is instance

def test_instance_context():
    with TensorProfiler() as prof:
        assert TensorProfiler.instance() is prof

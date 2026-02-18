from pttp import TensorProfiler


def test_catch_errors():
    finished = False
    try:
        with TensorProfiler():
            raise ValueError()

        finished = True
    finally:
        assert finished

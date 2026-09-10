from pttp import TensorProfiler


def test_catch_exception():
    finished = False
    try:
        with TensorProfiler(catch_exception=True):
            raise ValueError()

        finished = True
    finally:
        assert finished

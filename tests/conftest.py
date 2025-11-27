import math
import sys
import types


class _FakeArray(list):
    @property
    def shape(self):
        return (len(self),)

    def __truediv__(self, other):
        return _FakeArray([x / other for x in self])


def _fake_array(data, dtype=None):
    return _FakeArray(list(data))


def _fake_zeros(size, dtype=None):
    return _FakeArray([0.0] * int(size))


def _fake_dot(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b))


class _FakeLinalg:
    @staticmethod
    def norm(vec):
        return math.sqrt(sum(float(x) ** 2 for x in vec) or 1.0)


_fake_numpy = types.SimpleNamespace(
    array=_fake_array,
    zeros=_fake_zeros,
    dot=_fake_dot,
    linalg=_FakeLinalg(),
    float32=float,
)
sys.modules.setdefault("numpy", _fake_numpy)


class _FakeEmbeddings:
    def create(self, model=None, input=None):  # noqa: D401 - test stub
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[])])


class OpenAI:  # noqa: D401 - test stub
    def __init__(self, *_, **__):
        self.embeddings = _FakeEmbeddings()


sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=OpenAI))

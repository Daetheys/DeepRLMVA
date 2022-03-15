from replay_buffer import *


def test_init():
    buffer = BaseReplayBuffer(20)
    assert not buffer._memory
    assert buffer._maxlen == 20
    buffer = BaseReplayBuffer(42)
    assert buffer._maxlen == 42


def test_add():
    buffer = BaseReplayBuffer(5)
    buffer.add(0, 0, 0, 0, 0, True)
    assert buffer._memory
    for i in range(50):
        buffer.add(i, i, i, i, i, True)
    assert len(buffer._memory) == 5


def test_sample():
    buffer = BaseReplayBuffer(5)
    for i in range(5):
        buffer.add(i, i, i, i, i, True)
    for i in range(5):
        _ = buffer.sample()
    assert not buffer._memory


def test_sample_batch():
    buffer = ReplayBuffer(50)
    for i in range(50):
        buffer.add(i, i, i, i, i, True)
    for i in range(5):
        _ = buffer.sample_batch(10)
    assert not buffer._memory

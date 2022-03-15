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


def test_add_to_prioritized_replay_buffer():
    buffer = PrioritizedReplayBuffer(20)
    assert not buffer._memory
    assert not buffer._priority
    buffer.add(0, 0, 0, 0, 0, True, 0.1)
    assert buffer._memory
    assert buffer._priority
    for i in range(50):
        buffer.add(i, i, i, i, i, True, i)
    assert len(buffer._memory) == 20
    assert len(buffer._priority) == 20


def test_sample_from_prioritized_replay_buffer():
    buffer = PrioritizedReplayBuffer(20)
    for i in range(50):
        buffer.add(i, i, i, i, i, True, 0)
    buffer.add(50, 50, 50, 50, 50, True, 50)
    sample = buffer.sample()
    assert sample[4] == 50

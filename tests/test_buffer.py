from replay_buffer import *
import gym


def test_init():
    env = gym.make('CartPole-v1')
    buffer = ReplayBuffer(20,env)
    assert buffer.maxlen == 20
    buffer = ReplayBuffer(42,env)
    assert buffer.maxlen == 42


def test_add():
    env = gym.make('CartPole-v1')
    buffer = ReplayBuffer(5,env)
    buffer.add(0, 0, 0, 0, 0, 0, 0, 0)
    for i in range(50):
        buffer.add(i, i, i, i, i, i, i, i)
    assert buffer.size == 5


def test_sample():
    env = gym.make('CartPole-v1')
    buffer = ReplayBuffer(50,env)
    for i in range(50):
        buffer.add(i, i, i, i, i, i, i, i)
    for i in range(5):
        _ = buffer.sample_batch(10)

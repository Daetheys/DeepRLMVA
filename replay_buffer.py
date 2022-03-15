import collections
import jax
import jax.numpy as jnp
import numpy as np

Transition = collections.namedtuple("Transition",
                                    field_names=["obs_tm1", "action_tm1", "reward_t", "discount_t", "obs_t", "done"])


class BaseReplayBuffer:
    """Fixed-size buffer to store transition tuples."""

    def __init__(self, buffer_capacity: int):
        """Initialize a ReplayBuffer object.
        Args:
            buffer_capacity (int): number of samples the buffer can store
        """
        self._memory = list()
        self._maxlen = buffer_capacity

    def add(self, obs_tm1, action_tm1, reward_t, discount_t, obs_t, done):
        """Add a new transition to memory."""
        if len(self._memory) >= self._maxlen:
            self._memory.pop(0)  # remove first elem (oldest)

        transition = Transition(
            obs_tm1=obs_tm1,
            action_tm1=action_tm1,
            reward_t=reward_t,
            discount_t=discount_t,
            obs_t=obs_t,
            done=done)

        # convert every data into jnp array
        transition = jax.tree_map(jnp.array, transition)

        self._memory.append(transition)

    def sample(self):
        """Randomly sample a transition from memory."""
        assert self._memory, 'Replay buffer is unfilled. It is impossible to sample from it.'
        transition_idx = np.random.randint(0, len(self._memory))
        transition = self._memory.pop(transition_idx)

        return transition


class ReplayBuffer(BaseReplayBuffer):

    def sample_batch(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        assert len(self._memory) >= batch_size, 'Insufficient number of transitions in replay buffer.'
        all_transitions = [self.sample() for _ in range(batch_size)]

        stacked_transitions = []
        for i, _ in enumerate(all_transitions[0]):
            arrays = [t[i] for t in all_transitions]
            arrays = jnp.stack(arrays, axis=0)
            stacked_transitions.append(arrays)

        return Transition(*stacked_transitions)


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store transition tuples."""

    def __init__(self, buffer_capacity: int):
        """Initialize a ReplayBuffer object."""
        self._memory = list()
        self._priority = list()
        self._maxlen = buffer_capacity

    def add(self, obs_tm1, action_tm1, reward_t, discount_t, obs_t, done,
            priority):
        """Add a new transition to memory."""
        if len(self._memory) >= self._maxlen:
            self._memory.pop(0)  # remove first elem (oldest)
            self._priority.pop(0)

        transition = Transition(
            obs_tm1=obs_tm1,
            action_tm1=action_tm1,
            reward_t=reward_t,
            discount_t=discount_t,
            obs_t=obs_t,
            done=done)

        # convert every data into jnp array
        transition = jax.tree_map(jnp.array, transition)

        self._memory.append(transition)
        self._priority.append(priority)

    def sample(self):
        """Randomly sample a transition from memory."""
        assert self._memory, 'replay buffer is unfilled'
        assert len(self._memory) == len(self._priority)

        priority_sum = sum(self._priority)
        normalized_priority = [p / priority_sum for p in self._priority]

        transition_idx = np.random.choice(range(len(self._memory)),
                                          p=normalized_priority)

        transition = self._memory.pop(transition_idx)
        self._priority.pop(transition_idx)

        return transition

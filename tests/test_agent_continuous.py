import jax
import jax.numpy as jnp
import optax

from networks import actor_critic_net
from agent_continuous import policy, select_action, loss_actor_critic, update


def test_policy():

    net = actor_critic_net(3, mode='continuous')
    init, apply = net
    rng = jax.random.PRNGKey(42)
    states = jnp.zeros((15, 10))
    params = init(rng, states)

    mean, std, value = policy(params, apply, states, rng)
    assert mean.shape == (15, 3)
    assert std.shape == (15, 3)
    assert value.shape == (15, 1)


def test_select_action():
    net = actor_critic_net(3, mode='continuous')
    init, apply = net
    rng = jax.random.PRNGKey(42)
    state = jnp.zeros(10)
    params = init(rng, state)

    action, _ = select_action(params, apply, state, rng)
    assert action.shape == (3, )

    states = jnp.zeros((15, 10))
    params = init(rng, states)
    action, value = select_action(params, apply, states, rng)
    assert action.shape == (15, 3)
    assert value.shape == (15, 1)


def test_select_action_and_explore():
    net = actor_critic_net(3, mode='continuous')
    init, apply = net
    rng = jax.random.PRNGKey(42)
    states = jnp.zeros((15, 10))
    params = init(rng, states)

    action, value = select_action(params, apply, states, rng)
    assert action.shape == (15, 3)
    assert value.shape == (15, 1)


def test_loss_actor_critic():
    net = actor_critic_net(5, mode='continuous')
    init, apply = net
    rng = jax.random.PRNGKey(42)
    states = jnp.zeros((15, 10))
    new_observations = jnp.zeros((15, 10))
    params = init(rng, states)
    rewards = jnp.zeros((15, 1))
    discounts = jnp.zeros((15, 1))

    actions = jnp.zeros(15, dtype=int)
    clip_eps = 0.1
    adv = 0

    rng_old = jax.random.PRNGKey(10)
    params_old = init(rng_old, states)

    loss = loss_actor_critic(params, apply, states, rewards, discounts, new_observations, actions, clip_eps, params_old, adv, rng)


def test_update():

    net = actor_critic_net(5, mode='continuous')
    init, apply = net

    rng = jax.random.PRNGKey(42)
    rng_old = jax.random.PRNGKey(10)

    states = jnp.zeros((15, 10))
    params = init(rng, states)
    params_old = init(rng_old, states)

    actions = jnp.zeros(15, dtype=int)

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)
    clip_eps = 0.1
    rewards = jnp.zeros((15,))
    new_observations = jnp.zeros((15, 10))
    discounts = jnp.zeros((15,))
    advs = jnp.zeros((15,))

    batch = states, actions, rewards, new_observations, discounts, advs
    new_params, new_opt_state = update(apply, optimizer, params, batch, opt_state, clip_eps, params_old, rng)

    assert params['linear']['w'].shape == new_params['linear']['w'].shape

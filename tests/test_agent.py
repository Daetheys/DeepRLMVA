import jax
import jax.numpy as jnp
import optax

from networks import build_network, actor_critic_net
from agent import policy_discrete, select_action_discrete, loss_actor_critic, update


def test_policy_discrete():

    net = actor_critic_net(5)
    init, apply = net
    rng = jax.random.PRNGKey(42)
    state = jnp.zeros(10)
    params = init(rng, state)

    pi, value = policy_discrete(params, apply, state, rng)
    assert pi.shape == (5, )  # Probability distribution over actions

    state = jnp.zeros((15, 10))
    params = init(rng, state)
    pi, value = policy_discrete(params, apply, state, rng)
    assert pi.shape == (15, 5)


def test_select_action_discrete():
    net = actor_critic_net(5)
    init, apply = net
    rng = jax.random.PRNGKey(42)
    state = jnp.zeros(10)
    params = init(rng, state)

    action,_ = select_action_discrete(params, apply, state, rng)
    assert action in jnp.arange(5)


def test_loss_actor_critic():
    net = actor_critic_net(5)
    init, apply = net
    rng = jax.random.PRNGKey(42)
    states = jnp.zeros((15, 10))
    params = init(rng, states)

    target = 0
    actions = jnp.zeros(15, dtype=int)
    clip_eps = 0.1
    adv = 0

    rng_old = jax.random.PRNGKey(10)
    params_old = init(rng_old, states)

    loss = loss_actor_critic(params, apply, states, target, actions, clip_eps, params_old, adv, rng)
    assert loss == 0

    adv = 1
    loss = loss_actor_critic(params, apply, states, target, actions, clip_eps, params_old, adv, rng)
    assert loss == 1


def test_update():

    net = actor_critic_net(5)
    init, apply = net

    rng = jax.random.PRNGKey(42)
    rng_old = jax.random.PRNGKey(10)

    states = jnp.zeros((15, 10))
    params = init(rng, states)
    params_old = init(rng_old, states)

    values = jnp.zeros((15,))
    targets = jnp.zeros((15,))

    actions = jnp.zeros(15, dtype=int)

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)
    clip_eps = 0.1

    advs = jnp.zeros((15,))
    log_pi_old = jnp.zeros((15,))

    batch = states, actions, log_pi_old, values, targets, advs
    new_params, new_opt_state = update(params, apply, batch, optimizer, opt_state, clip_eps, params_old, rng)

    assert params['linear']['w'].shape == new_params['linear']['w'].shape
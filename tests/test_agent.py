import jax
import jax.numpy as jnp
import optax

from networks import build_network, simple_net
from agent import policy, select_action, loss_actor_critic, update


def test_policy():

    net = simple_net(2)
    init, apply = net
    rng = jax.random.PRNGKey(42)
    state = jnp.zeros(10)
    params = init(rng, state)

    value, pi = policy(params, apply, state, rng)
    assert pi.shape == 5  # Probability distribution over actions
    assert value.shape == (15, 10)


def test_select_action():
    net = build_network([5])
    init, fwd = net
    rng = jax.random.PRNGKey(42)
    x = jnp.zeros((15, 10))
    params = init(rng, x)
    apply = fwd(params=params, x=x, rng=rng)

    actions = jnp.arange(5)
    action = select_action(params, apply, actions, x, rng)
    assert action in actions
    assert type(action) == int


def test_loss_actor_critic():
    net = build_network([5])
    init, fwd = net
    rng = jax.random.PRNGKey(42)
    x = jnp.zeros((15, 10))
    params = init(rng, x)
    apply = fwd(params=params, x=x, rng=rng)

    target = jnp.zeros((15, 10))
    action = 0
    clip_eps = 0.1
    adv = 1

    rng_old = jax.random.PRNGKey(42)
    params_old = init(rng_old, x)

    value, pi = policy(params, apply, x)
    ratio = pi[action] / params_old[action]
    jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()

    loss = loss_actor_critic(params, apply, x, target, action, clip_eps, params_old, adv)

    assert loss


def test_update():
    states = jnp.zeros((15, 10))
    action = jnp.zeros((15, 10))

    # simple network
    net = build_network([5])
    init, fwd = net
    rng = jax.random.PRNGKey(42)
    params = init(rng, states)
    apply = fwd(params=params, x=states, rng=rng)

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)
    clip_eps = 0.1
    adv = 1

    value = jnp.zeros((5,10))
    target = jnp.zeros((5,10))

    log_pi_old = jnp.zeros(5)
    rng = jax.random.PRNGKey(30)
    params_old = init(rng, states)

    batch = states, action, log_pi_old, value, target, adv

    new_params, new_opt_state = update(params, apply, batch, optimizer, opt_state, clip_eps, params_old)

    assert new_opt_state.shape == opt_state.shape
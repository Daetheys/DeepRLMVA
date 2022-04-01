import jax
import jax.numpy as jnp
import optax
import haiku as hk

from networks import actor_net,value_net
from agent import policy, select_action, select_action_and_explore, loss_actor, loss_critic, update


def test_policy():

    net = actor_net(3)
    init, apply = net
    states = jnp.zeros((15, 10))
    rng = jax.random.PRNGKey(42)
    params = init(rng, states)

    mean, std = policy(params, apply, states)
    assert mean.shape == (15, 3)
    assert std.shape == (1, 3)


def test_select_action():
    net = actor_net(3)
    init, apply = net
    state = jnp.zeros(10)
    rng = jax.random.PRNGKey(42)
    params = init(rng, state)

    action, _ = select_action(params, apply, state,rng)
    assert action.shape == (3, )

    states = jnp.zeros((15, 10))
    params = init(rng, states)
    action, _ = select_action(params, apply, states,rng)
    assert action.shape == (15, 3)


def test_select_action_and_explore():
    net = actor_net(3)
    init, apply = net
    states = jnp.zeros((15, 10))
    rng = hk.PRNGSequence(42)
    params = init(next(rng), states)

    action, logp = select_action_and_explore(params, apply, states,rng)
    assert action.shape == (15, 3)
    assert logp.shape == (15, 1)


def test_loss_actor():
    net = actor_net(5)
    init, apply = net
    rng = hk.PRNGSequence(42)
    states = jnp.zeros((15, 10))
    new_observations = jnp.zeros((15, 10))
    params = init(next(rng), states)
    rewards = jnp.zeros((15, 1))
    discounts = jnp.zeros((15, 1))

    actions = jnp.zeros((15,5))
    clip_eps = 0.1
    adv = jnp.zeros((15,1))

    logp = jnp.zeros((15,1))

    loss = loss_actor(params, apply, states, discounts, actions, clip_eps, logp, adv)

def test_loss_critic():
    net = value_net()
    init, apply = net
    rng = jax.random.PRNGKey(42)
    states = jnp.zeros((15, 10))
    new_observations = jnp.zeros((15, 10))
    params = init(rng, states)

    actions = jnp.zeros((15,1))
    adv = jnp.zeros((15,1))
    values = jnp.zeros((15,1))
    clip_eps = 0.1

    loss = loss_critic(params, apply, states,adv,values)


def test_update():

    act_net = actor_net(1)
    actor_init, actor_apply = act_net

    val_net = value_net()
    value_init, value_apply = val_net

    states = jnp.zeros((15, 10))

    rng = jax.random.PRNGKey(42)
    actor_params = actor_init(rng, states)
    value_params = value_init(rng, states)

    actions = jnp.zeros((15,1))

    actor_optimizer = optax.adam(3e-4)
    actor_opt_state = actor_optimizer.init(actor_params)

    value_optimizer = optax.adam(3e-4)
    value_opt_state = value_optimizer.init(value_params)
    
    clip_eps = 0.1
    rewards = jnp.zeros((15,1))
    new_observations = jnp.zeros((15, 10))
    discounts = jnp.zeros((15,1))
    advs = jnp.zeros((15,1))
    logp = jnp.zeros((15,1))
    return_ = jnp.zeros((15,1))

    batch = states, actions, rewards, new_observations, logp, discounts, advs, return_
    new_policy_params,new_value_params,new_policy_opt_state,new_value_opt_state,policy_loss,value_loss,policy_grad_norm_mean,value_grad_norm_mean = update(actor_apply, value_apply, actor_optimizer, value_optimizer, actor_params, value_params, batch, actor_opt_state, value_opt_state, clip_eps)

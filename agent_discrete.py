import optax
import jax
import jax.numpy as jnp
from jax.random import choice
from jax.tree_util import tree_map


def policy(params, apply, states, rng):
    pi, value = apply(params, x=states, rng=rng)
    return pi, value


def select_action(params, apply, state, rng):
    pi, value = policy(params, apply, state, rng)
    actions = choice(rng, a=jnp.arange(pi.shape[0]), p=pi)
    return actions, value


def loss_actor_critic(params, apply, states, rewards, discounts, new_observations, actions, clip_eps, params_old, adv, rng):

    pi, value_predicted = policy(params, apply, states, rng)
    pi_old, _ = policy(params_old, apply, states, rng)

    _, value_predicted_next = policy(params, apply, new_observations, rng)
    targets = rewards + discounts * value_predicted_next

    loss_critic = jnp.square(value_predicted - targets).mean()

    pis = jax.vmap(lambda a, b: a[b])(pi, actions)
    pis_old = jax.vmap(lambda a, b: a[b])(pi_old, actions)
    ratio = pis / pis_old
    loss_actor = jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()

    return loss_critic + loss_actor


def update(apply, optimizer, params, batch, opt_state, clip_eps, params_old, rng, clip_grad=None):
    """
    :param params: Parameters of the model
    :param apply: forward function applied to the input samples
    :param batch: Batch containing N trajectories of size T and associated state, action,
    log probabilities, values, targets and advantages
    :param opt_state: optimizer state
    :param clip_eps: parameter used to clip the probability ratio between 1 - eps, 1 + eps
    :param params_old: parameters before the update
    :param adv: vector of advantage functions
    :param optimizer:
    :param rng: random generator
    """
    states, actions, rewards, new_observations, discounts, advs = batch

    grad_fn = jax.grad(loss_actor_critic)
    grads = grad_fn(params, apply, states, rewards, discounts, new_observations, actions, clip_eps, params_old, advs, rng)

    optimizer_policy,optimizer_value = optimizer
    
    #Policy update
    updates_policy, new_opt_state_policy = optimizer_policy.update(grads, opt_state)
    if not(clip_grad is None):
        clipped_updates_policy = tree_map(lambda g : jnp.clip(g,-clip_grad,clip_grad),updates_policy)
    else:
        clipped_updates_policy = updates_policy
    new_params = optax.apply_updates(params, clipped_updates_policy)

    #Value update
    updates_value, new_opt_state_value = optimizer_value.update(grads, opt_state)
    if not(clip_grad is None):
        clipped_updates_value = tree_map(lambda g : jnp.clip(g,-clip_grad,clip_grad),updates_value)
    else:
        clipped_updates_value = updates_value
    new_params = optax.apply_updates(new_params, clipped_updates_value)

    new_opt_state = new_opt_state_policy,new_opt_state_value

    return new_params, new_opt_state

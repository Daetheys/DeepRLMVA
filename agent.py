import optax
import jax
import jax.numpy as jnp
from jax.random import choice


DEFAULT_AGENT_CONFIG = {
    "clip_eps": 0.1
}


def policy(params, apply, state):
    value, pi = apply(params, state)
    return value, pi


def select_action(params, apply, actions, state, rng):
    _, pi = policy(apply, state, params)
    action = choice(rng, a=actions, p=pi)
    return action


def loss_actor_critic(params, apply, state, target, action, clip_eps, pi_old, adv):
    value_predicted, pi = policy(apply, state, params)
    loss_critic = jnp.square(value_predicted - target).mean()

    ratio = pi[action] / pi_old[action]
    loss_actor = jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()

    return loss_critic + loss_actor


def update(params, apply, batch, optimizer, opt_state, clip_eps, pi_old, adv):
    """
    :param params: Parameters of the model
    :param batch: Batch containing N trajectories of size T and associated state, action,
    log probabilities, values, targets and advantages
    :param opt_state: optimizer state
    :param clip_eps: parameter used to clip the probability ratio between 1 - eps, 1 + eps
    :param pi_old: vector of policy parameters before the update
    :param adv: vector of advantage functions
    """

    state, action, log_pi_old, value, target, a = batch

    grad_fn = jax.grad(loss_actor_critic)
    grads = grad_fn(params, apply, state, target, action, clip_eps, pi_old, adv)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state

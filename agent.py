import optax
import jax
import jax.numpy as jnp
from jax.random import choice, normal


def policy_discrete(params, apply, states, rng):
    pi, value = apply(params, x=states, rng=rng)
    return pi, value


def policy_continuous(params, apply, states, rng):
    (mean, std), value = apply(params, x=states, rng=rng)
    return mean, std, value


def select_action_discrete(params, apply, state, rng):
    pi, value = policy_discrete(params, apply, state, rng)
    actions = choice(rng, a=jnp.arange(pi.shape[0]), p=pi)
    return actions, value


def select_action_continuous(params, apply, state, exploration, rng):
    mu, sigma, value = policy_continuous(params, apply, state, rng)
    if exploration:
        action = mu + sigma * normal(key=rng, shape=mu.shape)
    else:
        action = mu
    return action, value


def loss_actor_critic(params, apply, states, target, actions, clip_eps, params_old, adv, rng):
    pi, value_predicted = policy_discrete(params, apply, states, rng)
    pi_old, _ = policy_discrete(params_old, apply, states, rng)
    loss_critic = jnp.square(value_predicted - target).mean()

    pis = jax.vmap(lambda a, b: a[b])(pi, actions)
    pis_old = jax.vmap(lambda a, b: a[b])(pi_old, actions)
    ratio = pis / pis_old
    loss_actor = jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()

    return loss_critic + loss_actor


def update(params, apply, batch, optimizer, opt_state, clip_eps, params_old, rng):
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

    states, actions, log_pi_olds, values, targets, advs = batch

    grad_fn = jax.grad(loss_actor_critic)
    grads = grad_fn(params, apply, states, targets, actions, clip_eps, params_old, advs, rng)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state


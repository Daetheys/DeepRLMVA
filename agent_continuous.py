import optax
import jax
import jax.numpy as jnp
from jax.random import normal
from utils import compute_probability_jitted


def policy(params, apply, states, rng):
    (mean, std), value = apply(params, x=states, rng=rng)
    return mean, std, value


def select_action(params, apply, state, rng):
    mu, sigma, value = policy(params, apply, state, rng)
    action = mu + sigma * normal(key=rng, shape=mu.shape)
    return action, value


def select_action_and_explore(params, apply, state, rng):
    mu, sigma, value = policy(params, apply, state, rng)
    action = mu
    return action, value


def loss_actor_critic(params, apply, states, rewards, discounts, new_observations, actions, clip_eps, params_old, adv, rng):

    mean, std, value_predicted = policy(params, apply, states, rng)
    mean_old, std_old, _ = policy(params_old, apply, states, rng)

    _, _,  value_predicted_next = policy(params, apply, new_observations, rng)
    targets = rewards + discounts * value_predicted_next
    loss_critic = jnp.square(value_predicted - targets).mean()

    pis = compute_probability_jitted(actions, mean, std)
    pis_old = compute_probability_jitted(actions, mean_old, std_old)

    ratio = pis / pis_old
    loss_actor = jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()

    return loss_critic + loss_actor


def update(apply, optimizer, params, batch, opt_state, clip_eps, params_old, rng):
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
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state


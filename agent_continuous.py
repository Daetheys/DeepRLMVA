import optax
import jax
import jax.numpy as jnp
from jax.random import normal
from utils import *
from jax.tree_util import tree_map
import numpy as np


def policy(params, apply, states, rng):
    mean, std = apply(params, x=states, rng=rng)
    return mean, std


def select_action_and_explore(params, apply, state, rng, cliprange=(-jnp.inf,jnp.inf)):
    mu, sigma = policy(params, apply, state, rng)

    rd = normal(key=rng, shape=mu.shape)
    
    action = mu + sigma * rd

    action = clip_action(scale_action(action,cliprange),cliprange)
    return action


def select_action(params, apply, state, rng, cliprange=(-jnp.inf,jnp.inf)):
    mu, sigma = policy(params, apply, state, rng)
    action = mu
    return clip_action(action,cliprange)

def check(p):
    if jnp.any(jnp.isnan(p)):
        assert False
    return p

def checkz(p):
    if jnp.any(jnp.isclose(p,0)):
        assert False
    return p

def loss_critic(value_params,value_apply,states,adv,rng):
    
    value_predicted = value_apply(value_params,x=states,rng=rng)

    targets = adv + value_predicted#(rewards + discounts * value_predicted_next[:,0])

    targets = jax.lax.stop_gradient(targets)

    loss_critic = jnp.square(value_predicted - targets).mean() #adv**2 ?

    return loss_critic

def loss_actor(policy_params,policy_apply,states,discounts,actions,clip_eps,policy_params_old,adv,rng):
    rng,srng = jax.random.split(rng)
    mean, std = policy(policy_params, policy_apply, states, srng)
    mean_old, std_old = policy(policy_params_old, policy_apply, states, srng)

    rng,srng = jax.random.split(rng)
    logpis = compute_logprobability_jitted(actions, mean, std)
    logpis_old = compute_logprobability_jitted(actions, mean_old, std_old)
    logpis_old = jax.lax.stop_gradient(logpis_old)
    
    log_ratio = logpis - logpis_old
    ratio = jnp.exp( log_ratio )[:,None]

    tree_map(check,logpis)
    tree_map(check,logpis_old)
    tree_map(check,ratio)

    loss_actor = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()

    return loss_actor

def update(policy_apply, value_apply, policy_optimizer, value_optimizer, policy_params, value_params, batch, policy_opt_state, value_opt_state, clip_eps, policy_params_old, rng, clip_grad=None):
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

    #Update critic
    value_grad_fn = jax.value_and_grad(loss_critic)
    rng,srng = jax.random.split(rng)
    value_loss,value_grads = value_grad_fn(value_params, value_apply, states, advs, srng)


    value_updates, new_value_opt_state = value_optimizer.update(value_grads,value_opt_state)
    new_value_params = optax.apply_updates(value_params,value_updates)

    #Update actor
    policy_grad_fn = jax.value_and_grad(loss_actor)
    rng,srng = jax.random.split(rng)
    policy_loss,policy_grads = policy_grad_fn(policy_params,policy_apply,states,discounts,actions,clip_eps,policy_params_old,advs,srng)

    policy_updates, new_policy_opt_state = policy_optimizer.update(policy_grads,policy_opt_state)
    new_policy_params = optax.apply_updates(policy_params,policy_updates)

    #Print losses
    print("Value loss :",value_loss,"- Actor loss :",policy_loss)

    return new_policy_params,new_value_params,new_policy_opt_state,new_value_opt_state












































                                               

def loss_actor_critic(policy_params, value_params, policy_apply, value_apply, states, rewards, discounts, new_observations, actions, clip_eps, params_old, adv, rng):

    mean, std = policy(policy_params, policy_apply, states, rng)
    value_predicted = value_apply(value_params,states,rng)
    
    mean_old, std_old = policy(params_old, apply, states, rng)

    #value_predicted_next = value_apply(value_params, new_observations, rng)
    targets = adv + value_predicted#(rewards + discounts * value_predicted_next[:,0])

    loss_critic = jnp.square(value_predicted - targets).mean()

    tree_map(check,actions)
    tree_map(check,mean)
    tree_map(check,std)
    tree_map(checkz,std)

    logpis = compute_logprobability_jitted(actions, mean, std)
    logpis_old = compute_logprobability_jitted(actions, mean_old, std_old)

    tree_map(check,logpis)
    tree_map(check,logpis_old)

    log_ratio = logpis - logpis_old
    #log_ratio = jnp.clip(log_ratio,-10,3)
    ratio = jnp.exp( log_ratio )[:,None]

    tree_map(check,ratio)

    loss_actor = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()

    loss = loss_critic + loss_actor

    return loss

def print_loss_actor_critic(params, apply, states, rewards, discounts, new_observations, actions, clip_eps, params_old, adv, rng):

    mean, std, value_predicted = policy(params, apply, states, rng)
    mean_old, std_old, _ = policy(params_old, apply, states, rng)

    _, _,  value_predicted_next = policy(params, apply, new_observations, rng)
    targets = adv + value_predicted#(rewards + discounts * value_predicted_next[:,0])

    print(targets[:5,0],value_predicted[:5,0])

    loss_critic = jnp.square(value_predicted - targets).mean()

    logpis = compute_logprobability_jitted(actions, mean, std)
    logpis_old = compute_logprobability_jitted(actions, mean_old, std_old)

    ratio = jnp.exp( logpis - logpis_old )[:,None]

    loss_actor = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()

    print(loss_critic,loss_actor)


def update2(apply, optimizer, params, batch, opt_state, clip_eps, params_old, rng, clip_grad=None):
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

    grad_fn = jax.value_and_grad(loss_actor_critic)
    val,grads = grad_fn(params, apply, states, rewards, discounts, new_observations, actions, clip_eps, params_old, advs, rng)
    #print('loss:',val)
    print_loss_actor_critic(params, apply, states, rewards, discounts, new_observations, actions, clip_eps, params_old, advs, rng)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    if not(clip_grad is None):
        clipped_updates = tree_map(lambda g : jnp.clip(g,-clip_grad,clip_grad),updates)
    else:
        clipped_updates = updates
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state


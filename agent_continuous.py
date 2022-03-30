import optax
import jax
import jax.numpy as jnp
from jax.random import normal
from utils import *
from jax.tree_util import tree_map
import numpy as np


def policy(params, apply, states, rng):
    mean, std = apply(params, x=states, rng=rng)
    return mean,std


def select_action_and_explore(params, apply, state, rng, cliprange=(-jnp.inf,jnp.inf)):
    mean,std = policy(params, apply, state, rng)

    action = mean + std * jax.random.normal(rng,mean.shape)

    #action = np.array([0.1]) #DEBUG
    logp = compute_logprobability_jitted(action[None], mean[None], std[None])
    
    #scaled_action = clip_action(scale_action(action,cliprange),cliprange)
    
    return action,logp


def select_action(params, apply, state, rng, cliprange=(-jnp.inf,jnp.inf)):
    mean, std = policy(params, apply, state, rng)
    
    action = mean
    logp = compute_logprobability_jitted(action[None], mean[None], std[None])
    
    #action = clip_action(scale_action(action,cliprange),cliprange)

    
    return action,logp


def check(p):
    assert not(jnp.any(jnp.isnan(p)))
    return p

def checkz(p):
    if jnp.any(jnp.isclose(p,0)):
        assert False
    return p

def loss_critic(value_params,value_apply,states,adv,values,rng):
    
    value_predicted = value_apply(value_params,x=states,rng=rng)

    targets = adv + values#(rewards + discounts * value_predicted_next[:,0])

    loss_critic = jnp.square(value_predicted - targets).mean() #adv**2 ?

    return loss_critic

def loss_actor(policy_params,policy_apply,states,discounts,actions,clip_eps,logpis_old,adv,kl_coeff,entropy_coeff,rng):
    rng,srng = jax.random.split(rng)
    mean, std = policy(policy_params, policy_apply, states, srng)

    logpis = compute_logprobability_jitted(actions, mean, std)
    
    log_ratio = logpis - logpis_old
    ratio = jnp.exp( log_ratio )[:,None]

    loss_actor = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv)

    kl = logpis.mean()

    entropy = jnp.log(std).mean()

    loss_actor = loss_actor.mean()

    return loss_actor #+ kl_coeff*kl - entropy_coeff*entropy

def update(policy_apply, value_apply, policy_optimizer, value_optimizer, policy_params, value_params, batch, policy_opt_state, value_opt_state, clip_eps, kl_coeff, entropy_coeff, rng):
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
    
    states, actions, rewards, new_observations, logp, discounts, advs, values = batch

    #Update critic
    value_grad_fn = jax.value_and_grad(loss_critic)
    rng,srng = jax.random.split(rng)
    value_loss,value_grads = value_grad_fn(value_params, value_apply, states, advs, values, srng)


    value_updates, new_value_opt_state = value_optimizer.update(value_grads,value_opt_state)
    new_value_params = optax.apply_updates(value_params,value_updates)

    #Update actor
    policy_grad_fn = jax.value_and_grad(loss_actor)
    rng,srng = jax.random.split(rng)
    policy_loss,policy_grads = policy_grad_fn(policy_params,policy_apply,states,discounts,actions,clip_eps,logp,advs,kl_coeff,entropy_coeff,srng)

    policy_updates, new_policy_opt_state = policy_optimizer.update(policy_grads,policy_opt_state)
    new_policy_params = optax.apply_updates(policy_params,policy_updates)

    #Return new policy / value parameters and optimizer states
    return new_policy_params,new_value_params,new_policy_opt_state,new_value_opt_state,policy_loss,value_loss












































                                               

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


import optax
import jax
import jax.numpy as jnp
from jax.random import normal
from utils import *
from jax.tree_util import tree_map
import numpy as np


def policy(params, apply, states):
    mean, logstd = apply(params, x=states)
    return mean,logstd


def select_action_and_explore(params, apply, state, rng, cliprange=(-jnp.inf,jnp.inf)):
    mean,logstd = policy(params, apply, state)
    std = jnp.exp(logstd)

    noise = jax.random.normal(next(rng),mean.shape)
    action = mean + std * noise

    action = jnp.tanh(action)

    #print(action,mean,std)
    #logp = compute_logprobability_jitted(action, mean, std)
    logp = compute_logprob_tanh(action,logstd,noise)
    
    #scaled_action = clip_action(scale_action(action,cliprange),cliprange)
    
    return action,logp


def select_action(params, apply, state, rng, cliprange=(-jnp.inf,jnp.inf)):
    mean, std = policy(params, apply, state)
    
    action = mean
    logp = compute_logprobability_jitted(action[None], mean[None], std[None])

    #action = jnp.tanh(action)
    
    #action = clip_action(scale_action(action,cliprange),cliprange)

    
    return action,logp

def loss_critic(value_params,value_apply,states,adv,values):
    
    value_predicted = value_apply(value_params,x=states)

    targets = adv + values#(rewards + discounts * value_predicted_next[:,0])

    loss_critic = jnp.square(value_predicted - targets).mean() #adv**2 ?

    return loss_critic

def loss_actor(policy_params,policy_apply,states,discounts,actions,clip_eps,logpis_old,adv,kl_coeff,entropy_coeff):
    mean, std = policy(policy_params, policy_apply, states)
    #actions_net_space = jnp.arctanh(actions)

    logpis = compute_logprobability_jitted(actions, mean, std)
    
    log_ratio = logpis - logpis_old
    ratio = jnp.exp( log_ratio )[:,None]

    loss_1 = ratio * adv
    loss_2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv

    loss_actor = -jnp.minimum(loss_1, loss_2)

    kl = logpis.mean()

    entropy = jnp.log(std).mean()

    loss_actor = loss_actor.mean()
    
    return loss_actor #+ kl_coeff*kl - entropy_coeff*entropy

def debug_loss_actor(policy_params,policy_apply,states,discounts,actions,clip_eps,logpis_old,adv,kl_coeff,entropy_coeff,rng):
    mean, std = policy(policy_params, policy_apply, states)
    #actions_net_space = jnp.arctanh(actions)

    logpis = compute_logprobability_jitted(actions, mean, std)
    
    log_ratio = logpis - logpis_old
    ratio = jnp.exp( log_ratio )[:,None]

    loss_1 = ratio * adv
    loss_2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv

    loss_actor = -jnp.minimum(loss_1, loss_2)

    kl = logpis.mean()

    entropy = jnp.log(std).mean()

    loss_actor = loss_actor.mean()
    
    to_debug = (actions,mean,std,logpis,loss_actor)
    
    return to_debug

def update(policy_apply, value_apply, policy_optimizer, value_optimizer, policy_params, value_params, batch, policy_opt_state, value_opt_state, clip_eps, kl_coeff, entropy_coeff):
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
    value_loss,value_grads = value_grad_fn(value_params, value_apply, states, advs, values)


    value_updates, new_value_opt_state = value_optimizer.update(value_grads,value_opt_state)
    new_value_params = optax.apply_updates(value_params,value_updates)

    #Update actor
    policy_grad_fn = jax.value_and_grad(loss_actor)
    policy_loss,policy_grads = policy_grad_fn(policy_params,policy_apply,states,discounts,actions,clip_eps,logp,advs,kl_coeff,entropy_coeff)

    to_debug = debug_loss_actor(policy_params,policy_apply,states,discounts,actions,clip_eps,logp,advs,kl_coeff,entropy_coeff,srng)

    policy_updates, new_policy_opt_state = policy_optimizer.update(policy_grads,policy_opt_state)
    new_policy_params = optax.apply_updates(policy_params,policy_updates)

    #Return new policy / value parameters and optimizer states
    return new_policy_params,new_value_params,new_policy_opt_state,new_value_opt_state,policy_loss,value_loss,to_debug












































                                               

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


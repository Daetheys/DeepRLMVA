import optax
import jax
import jax.numpy as jnp
from jax.random import normal
from utils import *
from jax.tree_util import tree_map,tree_leaves
import numpy as np


def policy(params, apply, states):
    """ Returns the action distribution associated to an input state """
    #Forwards the network
    mean, logstd = apply(params, x=states)
    return mean,logstd


def select_action_and_explore(params, apply, state, rng):
    """ Select action and explores (sample) """
    #Compute distribution from network
    mean,logstd = policy(params, apply, state)
    std = jnp.exp(logstd)

    #Sample action according to distribution
    normal = jax.random.normal(next(rng),mean.shape) #N(0,1)
    action = mean + std * normal

    #Clip action with tanh and computes according logprob
    action = jnp.tanh(action)
    logp = compute_logprob_tanh(action,logstd,normal)
    
    return action,logp


def select_action(params, apply, state, rng):
    """ Select action without exploring (returns mean) """
    #Compute distribution from network
    mean, std = policy(params, apply, state)

    #Returns the mean and clip with tanh
    action = mean
    action = jnp.tanh(action)

    #Logprob isn't used so their is no point is computing it
    return action,None


def loss_critic(value_params,value_apply,states,adv,values):
    """ Computes the critic loss for PPO """
    #Predict value
    value_predicted = value_apply(value_params,x=states)

    #Computes GAE targets
    targets = jax.lax.stop_gradient(adv + values) #stop gradient might not be necessary here

    loss_critic = jnp.square(value_predicted - targets).mean()

    return loss_critic


def loss_actor(policy_params,policy_apply,states,discounts,actions,clip_eps,logpis_old,adv):
    """ Computes the actor loss for PPO """
    #Get distributions of actions of the batch
    mean, logstd = policy(policy_params, policy_apply, states)

    #Computes the probability of these actions with the actual parameters
    normal = (jnp.arctanh(actions) - mean)/(jnp.exp(logstd)+1e-10)
    logpis = compute_logprob_tanh(actions,logstd,normal)

    #Computes the ratio for PPO
    log_ratio = logpis - logpis_old
    ratio = jnp.exp( log_ratio )

    #Standardize gae before computing the loss
    adv = (adv-adv.mean())/(adv.std()+1e-10)

    #Computes PPO Loss
    loss_1 = ratio * adv
    loss_2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    loss_actor = -jnp.minimum(loss_1, loss_2)

    loss_actor = loss_actor.mean()
    
    return loss_actor


def update(policy_apply, value_apply, policy_optimizer, value_optimizer, policy_params, value_params, batch, policy_opt_state, value_opt_state, clip_eps):
    """ Updates the networks on the given batch by gradient descent """
    
    states, actions, rewards, new_observations, logp, discounts, advs, values = batch

    #Update critic
    value_grad_fn = jax.value_and_grad(loss_critic)
    value_loss,value_grads = value_grad_fn(value_params, value_apply, states, advs, values)


    value_updates, new_value_opt_state = value_optimizer.update(value_grads,value_opt_state)
    new_value_params = optax.apply_updates(value_params,value_updates)

    #Update actor
    policy_grad_fn = jax.value_and_grad(loss_actor)
    policy_loss,policy_grads = policy_grad_fn(policy_params,policy_apply,states,discounts,actions,clip_eps,logp,advs)

    policy_updates, new_policy_opt_state = policy_optimizer.update(policy_grads,policy_opt_state)
    new_policy_params = optax.apply_updates(policy_params,policy_updates)

    #Gradient data processing
    policy_grad_norm = tree_map(jnp.linalg.norm,policy_grads)
    policy_grad_norm_mean = jnp.mean(jnp.array(tree_leaves(policy_grad_norm)))
    
    value_grad_norm = tree_map(jnp.linalg.norm,value_grads)
    value_grad_norm_mean = jnp.mean(jnp.array(tree_leaves(value_grad_norm)))
    
    #Return new policy / value parameters and optimizer states
    return new_policy_params,new_value_params,new_policy_opt_state,new_value_opt_state,policy_loss,value_loss,policy_grad_norm_mean,value_grad_norm_mean

import optax
import jax
import jax.numpy as jnp
from jax.random import choice
from jax.tree_util import tree_map


def policy(params, apply, states):
    pi = apply(params, x=states)
    return pi

batch_indexing = jax.vmap(lambda batch,idx : batch[idx],1)
def select_action(params, apply, state, rng):
    pi = policy(params, apply, state)
    action = choice(rng, a=jnp.arange(pi.shape[0]), p=pi)
    return action,pi[action]

def loss_critic(value_params,value_apply,states,adv,values,rng):
    
    value_predicted = value_apply(value_params,x=states)

    targets = adv + values#(rewards + discounts * value_predicted_next[:,0])

    loss_critic = jnp.square(value_predicted - targets).mean() #adv**2 ?

    return loss_critic

def loss_actor(policy_params,policy_apply,states,discounts,actions,clip_eps,logpis_old,adv,kl_coeff,entropy_coeff,rng):
    pi = policy(policy_params, policy_apply, states)

    logpis = jnp.log(pi)
    
    log_ratio = logpis - logpis_old
    ratio = jnp.exp( log_ratio )[:,None]

    loss_1 = ratio * adv
    loss_2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv

    loss_actor = -jnp.minimum(loss_1, loss_2)

    kl = logpis.mean()

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






























def loss_actor_critic(policy_params, apply, states, rewards, discounts, new_observations, actions, clip_eps, params_old, adv, rng):

    pi = policy(params, apply, states, rng)
    value = value_apply(value_params,states)
    pi_old, _ = policy(params_old, apply, states, rng)

    _, value_predicted_next = policy(params, apply, new_observations, rng)
    targets = rewards + discounts * value_predicted_next

    loss_critic = jnp.square(value_predicted - targets).mean()

    pis = jax.vmap(lambda a, b: a[b])(pi, actions)
    pis_old = jax.vmap(lambda a, b: a[b])(pi_old, actions)
    ratio = pis / pis_old
    loss_actor = jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()

    return loss_critic + loss_actor


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

    grad_fn = jax.grad(loss_actor_critic)
    grads = grad_fn(params, apply, states, rewards, discounts, new_observations, actions, clip_eps, params_old, advs, rng)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    if not(clip_grad is None):
        clipped_updates = tree_map(lambda g : jnp.clip(g,-clip_grad,clip_grad),updates)
    else:
        clipped_updates = updates
    new_params = optax.apply_updates(params, clipped_updates)

    return new_params, new_opt_state


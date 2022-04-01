import numpy as np
from statistics import mean
import jax
import jax.numpy as jnp

def calculate_gaes(rewards, values, done, new_value,gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    #Get next values
    next_values = values[1:]
    next_values = np.append(next_values, new_value)

    #Compute deltas
    deltas = [rew + gamma * next_val * (1-d) - val for rew, val, next_val,d in zip(rewards, values, next_values,done)]

    #Create np arrays
    rew = np.array(rewards)
    values = np.array(values)
    done = np.array(done)
    next_values = np.array(next_values)

    #Compute GAE (reverse order)
    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + gamma * decay * (1-done[i]) * gaes[-1])
    gaes = np.array(gaes)

    #Return GAES in the right order
    return np.array(gaes[::-1])

def rollout(select_action, env, nb_steps, replay_buffer, gamma, decay, policy_params, value_params, policy_apply, value_apply, rng, add_buffer = True, reward_scaling=1., mask_done=True, gae_std=True):
    """
    Performs a single rollout.
    Returns informations in a dict of shape (nb_steps, obs_shape)
    with mean reward and the mean timestep.
    """
    ### Create data storage
    traj_info = [[], [], [], [], [], []] # obs, act, reward, values, done
    obs = env.reset()

    #Initialize stats
    len_ep = []
    episode_length = 0
    reward_ep = 0
    rewards_ep = []
    step = 0

    ### Generate a trajectory of length nb_steps
    while step < nb_steps:# or not done:
        episode_length += 1
    
        act,logp = select_action(policy_params, policy_apply, obs[None], rng)  # Sample an action
        act = act[0]
        
        value = value_apply(value_params,x=obs) #Compute value

        next_obs, reward, done, i = env.step(np.array(act)) #Step the environment

        #Soft horizon
        mask = done
        if mask_done and episode_length == env._max_episode_steps:
            mask = False
        
        #Reward scaling
        reward *= reward_scaling

        reward_ep += reward #Log for stats

        ### Store Data
        for j, item in enumerate((obs, act, reward, value, mask, logp)):
          traj_info[j].append(item)

        #Loop and Reset if needed
        obs = next_obs
        step += 1
        if done:
            len_ep.append(episode_length)
            episode_length = 0
            rewards_ep.append(reward_ep)
            reward_ep = 0
            obs = env.reset()

    #Prepare data for the replay buffer
    traj_info = [jnp.array(x) for x in traj_info] #Set as array
    
    new_value = value_apply(value_params,x=next_obs) #Compute values
    gaes = calculate_gaes(traj_info[2], traj_info[3], traj_info[4],new_value,gamma=gamma,decay=decay)  # Compute GAES

    return_ = gaes + traj_info[3]

    if gae_std:
        gaes = (gaes - gaes.mean())/(gaes.std()+1e-6)

    #Add in the buffer (if requested)
    if add_buffer:
        for i in range(nb_steps-1):
            replay_buffer.add(traj_info[0][i], traj_info[1][i],traj_info[2][i], traj_info[0][i+1], traj_info[5][i], gamma*(1-traj_info[4][i]), gaes[i], return_[i])
        i += 1
        replay_buffer.add(traj_info[0][i], traj_info[1][i],traj_info[2][i], next_obs, traj_info[5][i], gamma*(1-traj_info[4][i]), gaes[i], return_[i])

    #Return stats to print them
    return dict(
        observations = traj_info[0],
        actions = traj_info[1],
        rewards = traj_info[2],
        gaes = traj_info[3]
        ), np.mean(rewards_ep), mean(len_ep) # obs, act, reward, gaes, mean_ep_reward, mean_ep_timestep 

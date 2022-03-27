import numpy as np
from statistics import mean

def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = values[1:]
    next_values = np.append(next_values, 0)  
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])

def rollout(select_action, env, nb_steps, replay_buffer, discount, params, apply, rng, add_buffer = True):
    """
    Performs a single rollout.
    Returns informations in a dict of shape (nb_steps, obs_shape)
    with mean reward and the mean timestep.
    """
    ### Create data storage
    traj_info = [[], [], [], []] # obs, act, reward, values
    obs = env.reset()
    len_ep = []
    episode_length = 0
    step = 0

    ### Generate a trajectory of length nb_steps
    while step < nb_steps or not done:
        episode_length += 1
        act, value = select_action(params, apply, obs, rng)  # Sample an action , to adapt
        next_obs, reward, done, i = env.step(act)

        ### Store Data
        for j, item in enumerate((obs, act, reward, value)):
          traj_info[j].append(item)

        obs = next_obs
        step += 1
        if done:
            len_ep.append(episode_length)
            episode_length = 0
            obs = env.reset()

    traj_info = [np.asarray(x) for x in traj_info]
    traj_info[3] = calculate_gaes(traj_info[2], traj_info[3])  # Calculate GAES
    
    if add_buffer: 
        for i in range(nb_steps-1):
                replay_buffer.add(traj_info[0][i], traj_info[1][i],traj_info[2][i], traj_info[0][i+1], discount, traj_info[3][i])
        
        replay_buffer.add(traj_info[0][nb_steps-1], traj_info[1][nb_steps-1],traj_info[2][nb_steps-1], traj_info[0][0], 0, traj_info[3][nb_steps-1])
    
    return dict(
        observations = traj_info[0],
        actions = traj_info[1],
        rewards = traj_info[2],
        gaes = traj_info[3]
        ), mean(traj_info[2]), mean(len_ep) # obs, act, reward, gaes, mean_ep_reward, mean_ep_timestep 
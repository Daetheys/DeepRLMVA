import numpy as np
from statistics import mean

def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])

def rollout(agent, env, nb_steps, replay_buffer):
    """
    Performs a single rollout.
    Returns training data in the shape (n_steps, observation_shape)
    and the cumulative reward.
    """
    ### Create data storage
    traj_info = [[], [], [], []] # obs, act, reward, values
    obs = env.reset()
    #traj_reward = 0
    len_ep = []
    episode_length = 0

    ### Generate a trajectory of length nb_steps
    for i in range(nb_steps):
        episode_length += 1
        act = agent.select_action(obs)  # Sample an action , to adapt
        value = agent.policy(obs)[0]
        next_obs, reward, done, i = env.step(act)

        ### Store Data
        for j, item in enumerate((obs, act, reward, value)):
          traj_info[j].append(item)
          
        replay_buffer.add(obs, act, reward, next_obs, done)
 

        obs = next_obs
        #traj_reward += reward
        if done:
            len_ep.append(episode_length)
            episode_length = 0
            obs = env.reset()

    traj_info = [np.asarray(x) for x in traj_info]
    traj_info[3] = calculate_gaes(traj_info[2], traj_info[3])  # Calculate GAES

    return dict(
        observations = traj_info[0],
        actions = traj_info[1],
        rewards = traj_info[2],
        gaes = traj_info[3]
        ), mean(traj_info[2]), mean(len_ep) # obs, act, reward, gaes, mean_ep_reward, mean_ep_timestep 
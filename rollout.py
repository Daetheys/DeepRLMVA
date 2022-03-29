import numpy as np
from statistics import mean
import jax

def calculate_gaes(rewards, values, done, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = values[1:]
    next_values = np.append(next_values, 0)
    #values = 0*values+1
    #next_values = next_values*0+1
    deltas = [rew + gamma * (1-d) * next_val - val for rew, val, next_val,d in zip(rewards, values, next_values,done)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * (1-done[i]) * gaes[-1])

    deltas = np.array(deltas)
    gaes = np.array(gaes)
        
    """print("----")
    print(gamma,decay)
    done = done[:500]
    for (m,r,v,d,g) in zip(done,rewards,values,deltas,gaes[::-1]):
        print(m,r,v,d,g)
    assert False"""

    return np.array(gaes[::-1])

def rollout(select_action, env, nb_steps, replay_buffer, gamma, decay, policy_params, value_params, policy_apply, value_apply, rng, add_buffer = True, reward_scaling=1.):
    """
    Performs a single rollout.
    Returns informations in a dict of shape (nb_steps, obs_shape)
    with mean reward and the mean timestep.
    """
    ### Create data storage
    traj_info = [[], [], [], [], []] # obs, act, reward, values, done
    obs = env.reset()
    #env.env.env.state = np.array([0.1,0.1])
    #obs,_,_,_ = env.step(np.array([0.1]))
    len_ep = []
    episode_length = 0
    reward_ep = 0
    rewards_ep = []
    step = 0

    ### Generate a trajectory of length nb_steps
    while step < nb_steps:# or not done:
        episode_length += 1
        
        rng,sub_rng = jax.random.split(rng)
        act = select_action(policy_params, policy_apply, obs, sub_rng)  # Sample an action , to adapt
        #act = np.array([0.1])
        rng,sub_rng = jax.random.split(rng)
        value = value_apply(value_params,x=obs,rng=sub_rng)
            
        next_obs, reward, done, i = env.step(act)

        reward_ep += reward

        reward *= reward_scaling

        ### Store Data
        for j, item in enumerate((obs, act, reward, value, done)):
          traj_info[j].append(item)

        obs = next_obs
        step += 1
        if done:
            len_ep.append(episode_length)
            episode_length = 0
            rewards_ep.append(reward_ep)
            reward_ep = 0
            obs = env.reset()
            #env.env.env.state = np.array([0.1,0.1])
            #obs,_,_,_ = env.step(np.array([0.1]))

    traj_info = [np.asarray(x) for x in traj_info]
    #print(traj_info[0])
    #print(traj_info[1])
    #print(traj_info[2])
    #assert False
    #print(traj_info[3])
    
    traj_info[3] = calculate_gaes(traj_info[2], traj_info[3], traj_info[4],gamma=gamma,decay=decay)  # Calculate GAES

    #print(traj_info[3][:,0].tolist())
    #assert False
    
    if add_buffer: 
        for i in range(nb_steps-1):
                replay_buffer.add(traj_info[0][i], traj_info[1][i],traj_info[2][i], traj_info[0][i+1], gamma*(1-done), traj_info[3][i])
        
        replay_buffer.add(traj_info[0][nb_steps-1], traj_info[1][nb_steps-1],traj_info[2][nb_steps-1], traj_info[0][0], gamma*(1-done), traj_info[3][nb_steps-1])

    
    return dict(
        observations = traj_info[0],
        actions = traj_info[1],
        rewards = traj_info[2],
        gaes = traj_info[3]
        ), np.mean(rewards_ep), mean(len_ep) # obs, act, reward, gaes, mean_ep_reward, mean_ep_timestep 

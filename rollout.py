import numpy as np
from statistics import mean
import jax
import jax.numpy as jnp

def calculate_gaes(rewards, values, done, new_value,gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    rewards = np.float32(rewards)
    done = np.float32(done)
    next_values = values[1:]
    next_values = np.append(next_values, new_value)
    #values = 0*values+1
    #next_values = next_values*0+1
    deltas = [rew + gamma * next_val * (1-d) - val for rew, val, next_val,d in zip(rewards, values, next_values,done)]

    rew = np.array(rewards)
    values = np.array(values)
    done = np.array(done)
    next_values = np.array(next_values)

    gaes = [deltas[-1]]
    c = 0
    for i in reversed(range(len(deltas)-1)):
        #print(deltas[i],gaes[-1],done[i],gamma*decay*(1-done[i])*gaes[-1])
        c += 1
        #assert c < 205
        gaes.append(deltas[i] + gamma * decay * (1-done[i]) * gaes[-1])
    #assert False

    deltas = np.array(deltas)
    print("delta",deltas[-10:].tolist())
    print("rew",rew[-10:].tolist())
    print("done",done[-10:].tolist())
    print("values",values[-10:,0].tolist())
    print("nvalues",next_values[-10:].tolist())
    assert False
    gaes = np.array(gaes)
    print(gaes[::-1][-50:])
        
    """print("----")
    print(gamma,decay)
    done = done[:500]
    for (m,r,v,d,g) in zip(done,rewards,values,deltas,gaes[::-1]):
        print(v.dtype)
        assert False
        print(m,r,v[0].item(),d,g)
    assert False"""

    return np.array(gaes[::-1])

def rollout(select_action, env, nb_steps, replay_buffer, gamma, decay, policy_params, value_params, policy_apply, value_apply, rng, add_buffer = True, reward_scaling=1.):
    """
    Performs a single rollout.
    Returns informations in a dict of shape (nb_steps, obs_shape)
    with mean reward and the mean timestep.
    """
    ### Create data storage
    traj_info = [[], [], [], [], [], []] # obs, act, reward, values, done
    obs = env.reset()

    """print('CHECKPOINT OBS')
    print(obs)
    assert False"""
    
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
    
        act,logp = select_action(policy_params, policy_apply, obs[None], rng)  # Sample an action , to adapt
        act = act[0]
        #rng,sub_rng = jax.random.split(rng)
        
        value = value_apply(value_params,x=obs)

        next_obs, reward, done, i = env.step(np.array(act))

        mask = done
        if episode_length == env._max_episode_steps:
            mask = False

        reward_ep += reward

        reward *= reward_scaling

        print('CHECKPOINT ROLLOUT')
        print(obs.tolist())
        print(act.item(),logp.item())
        print(next_obs.tolist(),reward,done)
        #assert False

        """if step > 205:
            assert False"""

        ### Store Data
        for j, item in enumerate((obs, act, reward, value, mask, logp)):
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

    print(traj_info[3][-10:].tolist())

    traj_info = [jnp.array(x) for x in traj_info]

    new_value = value_apply(value_params,x=next_obs)
    gaes = calculate_gaes(traj_info[2], traj_info[3], traj_info[4],new_value,gamma=gamma,decay=decay)  # Calculate GAES
    
    if add_buffer:
        for i in range(nb_steps-1):
            replay_buffer.add(traj_info[0][i], traj_info[1][i],traj_info[2][i], traj_info[0][i+1], traj_info[5][i], gamma*(1-traj_info[4][i]), gaes[i], traj_info[3][i])
        replay_buffer.add(traj_info[0][i], traj_info[1][i],traj_info[2][i], next_obs, traj_info[5][i], gamma*(1-traj_info[4][i]), gaes[i], traj_info[3][i])

    
    return dict(
        observations = traj_info[0],
        actions = traj_info[1],
        rewards = traj_info[2],
        gaes = traj_info[3]
        ), np.mean(rewards_ep), mean(len_ep) # obs, act, reward, gaes, mean_ep_reward, mean_ep_timestep 

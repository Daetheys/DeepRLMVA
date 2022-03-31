DEFAULT_AGENT_CONFIG = {
    "clip_eps": 0.1
}

DEFAULT_TRAINER_CONFIG = {
    "nb_fit_per_epoch":10*2048//64,
    "train_batch_size":64,
    "training_rollout_length":2048,
    "testing_rollout_length":1000,
    "policy_optimizer":"adam",
    "policy_learning_rate": 3e-4,
    "value_optimizer":"adam",
    "value_learning_rate": 3e-4,
    "kl_coeff":0.0,
    "entropy_coeff":0.0,
    "clip_eps":0.2,
    "clip_grad":10.,
    "reward_scale":1.,
    "seed": 0,
    "gamma":0.995,
    "decay":0.97,
    "replay_buffer_size":2048,
}

DEFAULT_REPLAY_BUFFER_CONFIG = {}

DEFAULT_ROLLOUT_CONFIG = {}


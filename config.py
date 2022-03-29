DEFAULT_AGENT_CONFIG = {
    "clip_eps": 0.1
}

DEFAULT_TRAINER_CONFIG = {
  "nb_fit_per_epoch":2048//64,
  "train_batch_size":64,
  "training_rollout_length":2048,
  "testing_rollout_length":1000,
  "policy_optimizer":"adam",
  "policy_learning_rate": 1e-3,
  "value_optimizer":"adam",
  "value_learning_rate": 5e-3,
  "clip_eps":0.2,
  "clip_grad":None,
  "reward_scale":1.,
  "seed": 42,
  "gamma":0.9,
  "decay":0.8
}

DEFAULT_REPLAY_BUFFER_CONFIG = {}

DEFAULT_ROLLOUT_CONFIG = {}


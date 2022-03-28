DEFAULT_AGENT_CONFIG = {
    "clip_eps": 0.1
}

DEFAULT_TRAINER_CONFIG = {
  "nb_fit_per_epoch":30,
  "train_batch_size":128,
  "training_rollout_length":4000,
  "testing_rollout_length":1000,
  "learning_rate": 3e-3,
  "clip_eps":0.3,
  "seed": 42,
  "gamma":0.99
}

DEFAULT_REPLAY_BUFFER_CONFIG = {}

DEFAULT_ROLLOUT_CONFIG = {}


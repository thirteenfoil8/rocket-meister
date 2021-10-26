from ray import tune
from rocket_gym import RocketMeister10
tune.run(
    "PPO", # reinforced learning agent
    name = "Training2",
    # to resume training from a checkpoint, set the path accordingly:
    # resume = True, # you can resume from checkpoint
    # restore = r'.\ray_results\Example\SAC_RocketMeister10_ea992_00000_0_2020-11-11_22-07-33\checkpoint_3000\checkpoint-3000',
    checkpoint_freq = 1000,
    checkpoint_at_end = True,
    local_dir = r'./ray_results/',
    config={
        "env": RocketMeister10,
        "num_workers": 1,
        "num_cpus_per_worker": 4,
        "env_config":{
            "max_steps": 1000,
            "export_frames": False,
            "export_states": False,
            # "reward_mode": "continuous",
            # "env_flipped": True,
            # "env_flipmode": True,
            }
        },
    stop = {
        "timesteps_total": 5_000_000,
        },
    )
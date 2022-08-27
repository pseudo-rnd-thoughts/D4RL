from gym.envs.registration import register

from d4rl import infos
from d4rl.gym_bullet import gym_envs

for agent in ["hopper", "halfcheetah", "ant", "walker2d"]:
    register(
        id=f"bullet-{agent}-v0",
        entry_point=f"d4rl.gym_bullet.gym_envs:get_{agent}_env",
        max_episode_steps=1000,
    )

    for dataset in ["random", "medium", "expert", "medium-expert", "medium-replay"]:
        env_name = f"bullet-{agent}-{dataset}-v0"
        register(
            id=env_name,
            entry_point=f"d4rl.gym_bullet.gym_envs:get_{agent}_env",
            max_episode_steps=1000,
            kwargs={
                "ref_min_score": infos.REF_MIN_SCORE[env_name],
                "ref_max_score": infos.REF_MAX_SCORE[env_name],
                "dataset_url": infos.DATASET_URLS[env_name],
            },
        )

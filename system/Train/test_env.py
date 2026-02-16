
try:
    from PentestGymEnv import PentestGymEnv
    env = PentestGymEnv()
    print("Environment created successfully")
    obs, _ = env.reset()
    print(f"Initial Observation: {obs}")
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step taken: Action={action}, Reward={reward}, Done={done}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

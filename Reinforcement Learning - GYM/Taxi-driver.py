import gym
import time

env = gym.make("Taxi-v3").env
print(
    "State space:", env.observation_space,
    "\nAction Space:", env.action_space
)
env.render()

encoded_state = env.encode(2, 3, 3, 2)  # Taxi row, column, passenger index, dest.
print("Encoded state is:", encoded_state)

# Transfer the state to environment
env.s = encoded_state
env.render()
env.reset()

# probability, next_state, reward, done
print((env.P[encoded_state]))

# %% 1 Episode
env.reset()

time_step = 0
total_reward = 0
list_visualization = []

while True:

    time_step += 1
    act = env.action_space.sample()

    next_state, reward, done, info = env.step(act)
    total_reward += reward

    list_visualization.append({
        "frame": env,
        "reward": reward,
        "Total reward": total_reward,
        "State": next_state,
        "Action": act,
    })

    # env.render()
    if done:
        break
# %% 25 Episode
total_reward_list = []
for i in range(25):
    env.reset()
    time_step = 0
    total_reward = 0
    list_visualization = []

    while True:
        time_step += 1
        act = env.action_space.sample()

        next_state, reward, done, info = env.step(act)
        total_reward += reward

        list_visualization.append({
            "frame": env,
            "reward": reward,
            "Total reward": total_reward,
            "State": next_state,
            "Action": act,
        })

        # env.render()
        if done:
            total_reward_list.append(total_reward)
            break

# print(list_visualization)
# %%
for i, frame in enumerate(list_visualization):
    print(
        frame["frame"].render(),
        "\nTime:", i+1,
        "State:", frame["State"],
        "Action:", frame["Action"],
        "Reward:", frame["reward"],
        "Total Reward:", frame["Total reward"]
        )
    # time.sleep(0.5)

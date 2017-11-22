"""
Complete the env.py
"""
from part3.env import ArmEnv
from part3.rl import DDPG

MAX_EPISODES = 500
MAX_EP_STEPS = 200

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method
rl = DDPG(a_dim, s_dim, a_bound)

# start training
for i in range(MAX_EPISODES):
    s = env.reset()
    for j in range(MAX_EP_STEPS):
        env.render()

        a = rl.choose_action(s)

        s_, r, done = env.step(a)

        rl.store_transition(s, a, r, s_)

        if rl.memory_full:
            # start to learn once has fulfilled the memory
            rl.learn()

        s = s_

# summary:
"""
env should have at least:
env.reset()
env.render()
env.step()

while RL should have at least:
rl.choose_action()
rl.store_transition()
rl.learn()
rl.memory_full
"""




"""
Plug a RL method to the framework, this method can be discrete or continuous.
This script is based on a continuous action RL. If you want to change to discrete RL like DQN,
please change the env.py and rl.py correspondingly.
"""
from part4.env import ArmEnv
from part4.rl import DDPG

MAX_EPISODES = 700
MAX_EP_STEPS = 200

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
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
        if done or j == MAX_EP_STEPS-1:
            print('Ep: %i | %s' % (i, '---' if not done else 'done'))
            break

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




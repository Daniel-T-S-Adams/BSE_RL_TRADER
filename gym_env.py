import gymnasium as gym
from environment import AuctionEnv, lob

# Register the gymnasium environment
gym.register(
    id='Auction-v0',
    entry_point='__main__:AuctionEnv',
)

env = gym.make('Auction-v0', lob=lob)
obs = env.reset()
print(obs)

done = False
while not done:
    action = env.action_space.sample()  # Random action selection
    obs, reward, done, trun, info = env.step(action)
    print('Observation:', obs)
    print('Reward:', reward)
    print('Done:', done)

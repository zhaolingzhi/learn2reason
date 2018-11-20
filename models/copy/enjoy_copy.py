import sys
import gym
import numpy as np

from baselines import deepq


def main():
    env = gym.make("Copy-v0")
    act = deepq.load(sys.argv[1])
    # for i in range(5):
    #     i = np.asarray([i])
    #     print(act(i,False))
    # sys.exit()

    # while True:
    # if True:
    print('input space', env.observation_space)
    ncorrect = 0
    for i in range(100): 
        obs, done = env.reset(), False
        episode_rew = 0
        nsteps = 0
        while not done:
            obs = np.asarray([obs])
            # print('obs', obs)
            act0 = act(obs,False)[0]
            # print('act0', act0)
            action=(act0//10, (act0%10)//5, act0%5)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
            nsteps += 1

        env.render()
        print("Episode reward", episode_rew)
        if done:
            if abs(nsteps - episode_rew) < 0.1:
                ncorrect += 1
    print('ncorrect', ncorrect)


if __name__ == '__main__':
    main()

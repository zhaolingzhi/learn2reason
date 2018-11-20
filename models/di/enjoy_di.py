import sys
import gym
import numpy as np

from baselines import deepq
import numpy as np

def main():
    env = gym.make("DuplicatedInput-v0")
    act = deepq.load(sys.argv[1])

    print('input space', env.observation_space)
    ncorrect = 0
    for i in range(100): 
    # for i in range(30): 
        print('index', i)
        obs, done = env.reset(), False
        episode_rew = 0
        nsteps = 0
        last = 0
        obs = obs*10+last
        obs = np.asarray([obs])
        while not done:
            print('obs', obs)
            act0 = act(obs,False)[0]
            action=(1, act0//50, (act0%50)//10)
            last = act0%10
            print('action', action, last)
            new_obs, rew, done, _ = env.step(action)
            new_obs = new_obs*10+last
            new_obs = np.asarray([new_obs])
            obs = new_obs
            episode_rew += rew
            nsteps += 1

        env.render()
        print("Episode reward", episode_rew)
        # print("target", env.get_target())
        if done:
            # if abs(len(env.get_target()) - episode_rew) < 0.1 and last == 1:
            if abs(len(env.get_target()) - episode_rew) < 0.1:
                ncorrect += 1
            else:
                print('error', i)
    print('ncorrect', ncorrect)


if __name__ == '__main__':
    main()

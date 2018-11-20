import sys
import gym
import numpy as np

from baselines import deepq
import numpy as np

def main():
    env = gym.make("SingleMultiplication-v0")
    act = deepq.load(sys.argv[1])

    print('input space', env.observation_space)
    ncorrect = 0
    for i in range(100): 
    # for i in range(36): 
            # continue
        print('index', i)
        obs, done = env.reset(), False
        episode_rew = 0
        nsteps = 0
        obs = env.get_inputs()
        carry = 0
        obs = list(obs)
        obs.append(carry)
        obs = np.asarray(obs)
        while not done:
            act0 = act(obs[None],False)[0]

            print('obs', obs)
            print('act', act0)

            action=(1, 1, act0//3)
            # action=(1, act0//9, (act0%9)//3)
            carry = act0%3

            # act0, carry = get_result(obs)
            # action=(1, 1, act0)

            obs, rew, done, _ = env.step(action)
            # print('reward', rew)
            new_obs = env.get_inputs()
            new_obs = list(new_obs)
            new_obs.append(carry)
            new_obs = np.asarray(new_obs)
            obs = new_obs
            episode_rew += rew
            nsteps += 1

        env.render()
        print("Episode reward", episode_rew, 'nsteps', nsteps)
        # print("target", env.get_target())
        if done:
            if abs(len(env.get_target()) - episode_rew) < 0.1 and carry == 0:
            # if abs(len(env.get_target()) - episode_rew) < 0.1:
                ncorrect += 1
            elif nsteps >= 200:
                ncorrect += 1
            else:
                print('error', i)
    print('ncorrect', ncorrect)


if __name__ == '__main__':
    main()

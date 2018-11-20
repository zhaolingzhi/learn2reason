import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule
from baselines.deepq.simple import ActWrapper
import sys
# version = sys.argv[1]
version = '2'

def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def make_obs_ph(name):
    return ObservationInput(env.observation_space, name=name)


if __name__ == '__main__':
    with U.make_session(8):
        # Create the environment
        env = gym.make("Copy-v"+version)
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=model,
            # num_actions=env.action_space.n,
            num_actions=2*2*5,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )

        act_params = {
            'make_obs_ph': lambda name: ObservationInput(env.observation_space, name=name),
            'q_func': model,
            'num_actions': 2*2*5,
        }

        sact = ActWrapper(act, act_params)

        # act=(act/10, (act%10)/5, act%5)
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        curr_steps = 0
        for t in itertools.count():
            # Take action and update exploration to the newest value
            obs = np.asarray([obs])
            # print('act', act)
            # print('obs', obs, )
            act0 = act(obs, update_eps=exploration.value(t))[0]
            action=(act0//10, (act0%10)//5, act0%5)
            # print('action', action)
            # action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, act0, rew, new_obs, float(done))
            # replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            curr_steps += 1
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 100:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    # print('obses', obses_t.shape, obses_tp1.shape)
                    obses_t = obses_t[:,0]
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    solved = True
                    for i in range(5):
                        i = np.asarray([i])
                        if act(i,False) != 15 + i:
                            solved = False
                    if solved == True:
                        print('finished!', 'steps', t)
                        sact.save("copy_model.pkl")
                        break
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()
                    # save model
                    print("Saving model to cartpole_model.pkl")
                    sact.save("copy_model.pkl")

            # if done and len(episode_rewards) % 10 == 0:
                # logger.record_tabular("steps", t)
                # logger.record_tabular("current steps", curr_steps)
                # logger.record_tabular("episodes", len(episode_rewards))
                # logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                # logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                # logger.dump_tabular()

            if done:
                curr_steps = 0
            # if len(episode_rewards) > 800:
            #     sact.save("copy_model0.pkl")
            #     break

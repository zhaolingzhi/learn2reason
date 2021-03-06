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
import numpy as np

train_len = 36

def test_model(env, act):
    ncorrect = 0
    for i in range(train_len): 
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
            action=(1, 1, act0//3)
            carry = act0%3
            obs, rew, done, _ = env.step(action)
            new_obs = env.get_inputs()
            new_obs = list(new_obs)
            new_obs.append(carry)
            new_obs = np.asarray(new_obs)
            obs = new_obs
            episode_rew += rew
            nsteps += 1

        if done:
            if abs(len(env.get_target()) - episode_rew) < 0.1 and carry == 0:
            # if abs(len(env.get_target()) - episode_rew) < 0.1:
                ncorrect += 1
    print('ncorrect', ncorrect)
    return ncorrect == train_len



def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def make_obs_ph(name):
    return ObservationInput(env.observation_space, name=name)


if __name__ == '__main__':

    with U.make_session(8):
        # Create the environment
        env = gym.make("SingleMultiplication-v2")

        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=model,
            # num_actions=env.action_space.n,
            num_actions=1*1*3*3,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-3),
        )

        act_params = {
            'make_obs_ph': lambda name: ObservationInput(env.observation_space, name=name),
            'q_func': model,
            'num_actions': 1*1*3*3,
        }

        sact = ActWrapper(act, act_params)

        # act=(act/10, (act%10)/5, act%5)
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.01)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        print('obs shape', env.observation_space)
        episode_rewards = [0.0]
        obs = env.reset()
        curr_steps = 0
        carry = 0
        obs = env.get_inputs()
        obs = list(obs)
        obs.append(carry)
        obs = np.asarray(obs)
        for t in itertools.count():
            # Take action and update exploration to the newest value
            # print('act', act)
            # print('obs', obs, )
            act0 = act(obs[None], update_eps=exploration.value(t))[0]
            # always moves right
            action=(1, 1, act0//3)
            carry = act0%3
            # print('action', action)
            # action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            new_obs = env.get_inputs()
            new_obs = list(new_obs)
            new_obs.append(carry)
            new_obs = np.asarray(new_obs)
            # Store transition in the replay buffer.

            if done and carry != 0:
                rew = -0.5
                carry = 0

            replay_buffer.add(obs, act0, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            curr_steps += 1

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    # print('obses', obses_t.shape, obses_tp1.shape)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 100 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("current steps", curr_steps)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if done and len(episode_rewards) % 1000 == 0:
                # test model
                succ = test_model(env, act)

                # save model
                print("Saving model to cartpole_model.pkl")
                sact.save("sm_model5.pkl")
                if succ == True:
                    break

            if done:
                obs = env.reset()
                obs = env.get_inputs()
                obs = list(obs)
                obs.append(carry)
                obs = np.asarray(obs)
                episode_rewards.append(0)

            if done:
                curr_steps = 0

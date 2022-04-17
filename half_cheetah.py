#!/usr/bin/env python
import os
import statistics
from datetime import timedelta
from time import sleep, time
import gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv


def train(env, timesteps):
    train_start = time()

    model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log='Tensorboard_Logs')

    model_name = f'DDPG-{int(timesteps/1000)}k'
    model.learn(total_timesteps=timesteps,
                tb_log_name=model_name,
                eval_env=env,
                eval_freq=10000,
                n_eval_episodes=50,
                eval_log_path=os.path.join('Training_Logs', model_name))

    # model.save(os.path.join('Saved_Models', f'half_cheetah_ddpg-{int(timesteps/1000)}k'))

    train_duration = time() - train_start
    print(f'Training completed in {timedelta(seconds=train_duration)}')

    return model, model_name


def evaluate(env, model_name=None):
    # Load model from file
    if model_name is None:
        models = list(os.listdir('Training_Logs'))
        for i, model in enumerate(models):
            print(f'{i+1}. {model}')
        option = int(input('Which model? '))
        model_name = models[option - 1]
        filepath = os.path.join('Training_Logs', models[option - 1], 'evaluations.npz')
    else:
        filepath = os.path.join('Training_Logs', model_name, 'evaluations.npz')

    # === Training Evaluation === #
    # Used npzviewer to determine keys of npz file (`pip install npzviewer`)
    # In this case, they are `ep_lengths`, `results`, `timesteps`
    with np.load(filepath) as data:
        print('\n\033[4mTraining Evaluation\033[0m')
        print('{:<12}{:<10}{:<10}'.format('Timesteps', 'Average', 'Std'))
        for i in range(len(data['timesteps'])):
            timesteps = data['timesteps'][i]
            average = round(sum(data['results'][i]) / len(data['results'][i]), 2)
            stdev = round(statistics.pstdev(data['results'][i]), 2)
            print('{:<12}{:<10}{:<10}'.format(timesteps, average, stdev))

    # === Best Model Evaluation === #
    model = DDPG.load(os.path.join('Training_Logs', model_name, 'best_model'), env=env)
    episodes = 50
    scores = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            score += reward
        scores.append(float(score))

    print('\n\033[4mBest Model Evaluation\033[0m')
    # Doing this because for some reason `import numpy as np` is overriding `builtin.sum()`
    average = round(sum(scores) / len(scores), 2)
    stdev = round(statistics.pstdev(scores), 2)
    print('{:<10}{:<10}'.format('Average', 'Std'))
    print('{:<10}{:<10}'.format(average, stdev))


def render(env, model=None):
    # Load model from file
    if model is None:
        models = list(os.listdir('Training_Logs'))
        for i, model in enumerate(models):
            print(f'{i+1}. {model}')
        option = int(input('Which model? '))
        filepath = os.path.join('Training_Logs', models[option - 1], 'best_model.zip')
        model = DDPG.load(filepath, env=env)

    obs = env.reset()
    done = False
    env.render()
    sleep(2)

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)


if __name__ == '__main__':
    env = gym.make('HalfCheetah-v2')
    env = DummyVecEnv([lambda: env])

    option = input('Train, evaluate, or render [T/e/r]? ').lower()

    if option.startswith('t') or not len(option):
        timesteps = int(input('How many timesteps (in thousands)? '))
        model, model_name = train(env, timesteps * 1000)

        eval_option = input('Evaluate [Y/n]? ').lower()
        if eval_option == 'y' or not len(eval_option):
            evaluate(env, model_name)

            render_option = input('Render [y/N]? ').lower()
            if render_option == 'y':
                render(env, model)

    elif option.startswith('e'):
        evaluate(env, model_name=None)

    elif option.startswith('r'):
        render(env, model=None)

    env.close()

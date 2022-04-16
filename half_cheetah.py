#!/usr/bin/env python
import os
from datetime import timedelta
from time import time, sleep

import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


def load_model_from_file(env):
    models = [file[:-4] for file in os.listdir('Saved_Models')]
    for i, model in enumerate(models):
        print(f'{i+1}. {model}')
    option = int(input('Which model? '))
    return DDPG.load(os.path.join('Saved_Models', models[option - 1]), env=env)


def train(env, timesteps):
    train_start = time()

    model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log='Training_Logs')
    model.learn(total_timesteps=timesteps)
    model.save(os.path.join('Saved_Models', f'half_cheetah_ddpg-{int(timesteps/1000)}k'))

    train_duration = time() - train_start
    print(f'Training completed in {timedelta(seconds=train_duration)}')

    return model


def evaluate(env, model=None):
    # Load model from file
    if model is None:
        model = load_model_from_file(env)


    episodes = 5
    for episode in range(episodes):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            # env.render()
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            score += reward
        print(f'Episode {episode+1} finished with score {score}')
    env.close()


def render(env, model=None):
    # Load model from file
    if model is None:
        model = load_model_from_file(env)

    obs = env.reset()
    done = False
    score = 0
    env.render()
    sleep(2)

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward


if __name__ == '__main__':
    env = gym.make('HalfCheetah-v2')
    env = DummyVecEnv([lambda: env])

    # Env spaces are both boxes
    # print(env.observation_space)
    # print(env.action_space)

    option = input('Train, evaluate, or render [T/e/r]? ').lower()

    if option.startswith('t') or not len(option):
        timesteps = int(input('How many timesteps? '))
        model = train(env, timesteps)

        option = input('Evaluate [Y/n]? ').lower()
        if option == 'y' or not len(option):
            evaluate(env, model)

            option = input('Render [Y/n]? ').lower()
            if option == 'y' or not len(option):
                render(env, model)

    elif option.startswith('e'):
        evaluate(env, model=None)

    elif option.startswith('r'):
        render(env, model=None)

    env.close()

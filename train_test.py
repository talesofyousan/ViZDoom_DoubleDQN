#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import tensorflow as tf
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
from tqdm import tqdm
import replay_memory
import network_double
import agent_dqn
import math
from helper import make_gif, set_imageio

# Q-learning settings
learning_rate = 0.0025
discount_factor= 0.99
resolution = (30, 45, 1)
n_epoch = 2
steps_per_epoch = 100
testepisodes_per_epoch = 5
config_file_path = "./config/simpler_basic.cfg"
model_path = "./model/"
freq_copy = 30
frame_repeat = 12
capacity = 10**4
batch_size = 64

sleep_time = 0.028

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    #game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

def exploration_rate(epoch):
    """# Define exploration rate change over time"""
    start_eps = 1.0
    end_eps = 0.1
    const_eps_epochs = 0.1 * n_epoch  # 10% of learning time
    eps_decay_epochs = 0.6 * n_epoch  # 60% of learning time

    if epoch < const_eps_epochs:
        return start_eps
    elif epoch < eps_decay_epochs:
        # Linear decay
        return start_eps - (epoch - const_eps_epochs) / \
                       (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
    else:
        return end_eps

def save_gif(game, agent, id):
    print("Saving gif file")

    images = []
    for step in range(testepisodes_per_epoch):

        game.new_episode()
        while not game.is_episode_finished():
            buff = game.get_state().screen_buffer
            best_action_index = agent.get_best_action(buff)

            images.append(buff)

            game.make_action(actions[best_action_index])

            for _ in range(frame_repeat):
                game.advance_action()

        print("total:", game.get_total_reward())

    make_gif(np.array(images),"./gifs/test%04d.gif"%(id),duration=len(images)*0.1,true_image=True,salience=False)


if __name__=="__main__":
    game = initialize_vizdoom(config_file_path)

    print("learning rate: %f" % learning_rate)
    print("discount_factor %f" % discount_factor)
    print("resolution:",resolution)
    print("frame_repeat: %d" % frame_repeat)
    print("capacity:",capacity)
    print("barch_size: %d" % batch_size)
    print("screen_format:",game.get_screen_format())
    n_actions = game.get_available_buttons_size()
    actions = np.eye(n_actions,dtype=np.int32).tolist()

    replay_memory = replay_memory.ReplayMemory(resolution,capacity)

    session = tf.Session()
    network = network_double.network_simple(session,resolution,n_actions,learning_rate)

    session.run(tf.global_variables_initializer())

    agent = agent_dqn.agent_dqn(network,replay_memory,actions,resolution,discount_factor,learning_rate,frame_repeat,batch_size, freq_copy)

    for epoch in range(n_epoch):

        print("Epoch %d \n -----" % (epoch))
        print("Training Phase")
        train_episodes_finished = 0
        train_scores = []
        total_train_scores = []

        if epoch == 0:
            agent.save_model(model_path+"%04d"%(epoch)+".ckpt")
        elif epoch == 5:
            agent.save_model(model_path+"%04d"%(epoch)+".ckpt")
        elif epoch == 9:
            agent.save_model(model_path+"%04d"%(epoch)+".ckpt")
        else:
            pass

        game.new_episode()

        for step in tqdm(range(steps_per_epoch)):
            if game.is_player_dead():
                game.respawn_player()
            
            agent.perform_learning_step(game,epoch,epoch*steps_per_epoch+step,exploration_rate(epoch))

            if step % 10 == 0:
                score = agent.reward_repeat
                train_scores.append(score)

            if game.is_episode_finished():
                score = game.get_total_reward()
                total_train_scores.append(score)
                game.new_episode()
                train_episodes_finished += 1

        print("%d training episodes played." % train_episodes_finished)

        train_scores = np.array(train_scores)
        total_train_scores = np.array(total_train_scores)
        print("Results: mean: %.1f(+-)%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
        print("Total Results: mean %.1f(plusminus)%.1f," %(total_train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % total_train_scores.min(), "max: %.1f," % total_train_scores.max())

    game.close
"""
    print("Test Phase")

    game.close()
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    test_scores=[]
    for step in range(testepisodes_per_epoch):

        game.new_episode()
        while not game.is_episode_finished():
            best_action_index = agent.get_best_action(game.get_state().screen_buffer)

            game.make_action(actions[best_action_index])
            sleep(sleep_time)

            for _ in range(frame_repeat):
                game.advance_action()

        print("total:", game.get_total_reward())
"""

import numpy as np
import skimage.color, skimage.transform
from random import sample, randint, random

class agent_dqn(object):

    def __init__(self,q_network, replay_buff, actions, resolution, discount_factor, learning_rate, frame_repeat, batch_size, freq_copy):

        self.q_network = q_network
        self.replay_buff = replay_buff
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.replay_buff = replay_buff
        self.actions = actions
        self.frame_repeat = frame_repeat
        self.batch_size = batch_size
        self.freq_copy = freq_copy

        self.reward_repeat = 0

    def preprocess(self,img):
        if len(img.shape) == 3:
            img = img.transpose(1,2,0)

        img = skimage.transform.resize(img, self.resolution)
        img = img.astype(np.float32)
        return img

    def perform_learning_step(self,game,epoch,global_step, eps):
    
        s1 = self.preprocess(game.get_state().screen_buffer)

        if random() <= eps:
            a = randint(0, len(self.actions) - 1)
        else:
            # Choose the best action according to the network.
            a = self.q_network.get_best_action(np.array([s1]))
        
        reward = game.make_action(self.actions[a],self.frame_repeat)
        self.reward_repeat = reward

        isterminal = game.is_episode_finished()
        s2 = self.preprocess(game.get_state().screen_buffer) if not isterminal else None

        # Remember the transition that was just experienced.
        self.replay_buff.add_transition(s1, a, s2, isterminal, reward)

        if self.replay_buff.size > self.batch_size:
            s1, a, s2, isterminal, r = self.replay_buff.get_sample(self.batch_size)

            q2 = self.q_network.get_q_target_values(s2)
            target_q = self.q_network.get_q_values(s1)

            target_q[np.arange(target_q.shape[0]),a] = r + self.discount_factor * (1-isterminal) * q2
            
            self.q_network.learn(s1, target_q, reward, global_step)
        
        if global_step % self.freq_copy == 0:
            self.q_network.copy_params()
        
        if global_step % 10 == 0:
            game.new_episode()
            while not game.is_episode_finished():
                screen_buff = self.preprocess(game.get_state().screen_buffer)
                action = self.q_network.get_best_action(np.array([screen_buff]))
                game.make_action(self.actions[action],self.frame_repeat)
            
            self.q_network.write_total_reward(game.get_total_reward(),global_step)
    
    def get_best_action(self, img):
        state = self.preprocess(img)
        return self.q_network.get_best_action(state)

    def restore_model(self, model_path):
        self.q_network.restore_model(model_path)

    def save_model(self, model_path):
        self.q_network.save_model(model_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
from collections import deque
import numpy as np
import cv2
REPLAY_SIZE = 2000
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
BATCH_SIZE = 16
GAMMA = 0.9
class NET(nn.Module):
    def __init__(self, observation_height, observation_width, action_space) -> None:
        super(NET, self).__init__()
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        self.relu = nn.ReLU()
        self.net = nn.Sequential( 
            nn.Conv2d(1, 32, kernel_size=[5,5],stride=1,padding='same'), 
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=[5,5],stride=1,padding='same'), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(int((self.state_w/4) * (self.state_h/4) * 64), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.action_dim)

    def forward(self,x):
        x = self.net(x)
        x = x.reshape(-1, int((self.state_w/4) * (self.state_h/4) * 64))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN(object):
    def __init__(self,observation_height, observation_width, action_space, model_file,log_file):
        self.model_file = model_file
        self.target_net = NET(observation_height,observation_width,action_space)
        self.target_net.to("mps")
        self.eval_net = NET(observation_height,observation_width,action_space)
        self.eval_net.to("mps")
        self.replay_buffer = deque()
        self.epsilon = INITIAL_EPSILON
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.action_dim = action_space

    def choose_action(self, state):
        # the output is a tensor, so the [0] is to get the output as a list
        # use epsilon greedy to get the action
        if random.random() <= self.epsilon:
            # if lower than epsilon, give a random value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            # print("choose_action")
            # print(state.shape)
            Q_value = self.eval_net(state)
            # if bigger than epsilon, give the argmax value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            # print("Q_value: ",Q_value.shape)
            # print("Q_value: ",Q_value)
            return torch.argmax(Q_value)
    def store_data(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
    def train(self):
        self.target_net.to("mps")
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        y_batch = []
        # print(next_state_batch.shape)
        Q_value_batch = self.target_net(torch.tensor(next_state_batch).to(dtype=torch.float).to("mps"))
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            # see if the station is the final station
            if done:
                y_batch.append(reward_batch[i])
            else:
                # the Q value caculate use the max directly
                y_batch.append(reward_batch[i] + GAMMA * torch.max(Q_value_batch[i]))
        # 假设 self.Q_value 是模型输出的Q值，self.action_input 是动作的one-hot编码表示
        # print(Q_value_batch.shape)
        # print(Q_value_batch.dtype)
        action_batch = torch.tensor(action_batch).to(dtype=torch.float).to("mps")
        # print(action_batch.shape)
        # print(action_batch.dtype)

        Q_action = torch.sum(Q_value_batch * action_batch, dim=1)
        y_batch = torch.tensor(y_batch).to(dtype=torch.float).to("mps")
        print(y_batch.shape)
        loss = self.loss(Q_action, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.target_net.state_dict(), self.model_file)
    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())


import cv2
from grabscreen import grab_screen
import time
import directkeys
from restart import restart


DQN_model_path = "model_gpu"
DQN_log_path = "logs_gpu/"
WIDTH = 96
HEIGHT = 88
window_size = (320,100,704,452)#384,352  192,176 96,88 48,44 24,22
# station window_size
blood_window = (60,91,280,562)
# used to get boss and self blood

action_size = 10
# action[n_choose,j,k,m,r]
# j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing

EPISODES = 3000
big_BATCH_SIZE = 16
UPDATE_STEP = 50
# times that evaluate the network
num_step = 0
# used to save log graph
target_step = 0
# used to update target Q network
paused = True
# used to stop training
def boss_blood_count(boss_gray):
    boss_blood = 0
    for boss_bd_num in boss_gray[10]:
    # boss blood gray pixel 65~75
    # 血量灰度值65~75 
        if boss_bd_num > 143 and boss_bd_num < 157:
            boss_blood += 2
    return boss_blood
def self_blood_count(self_gray):
    self_blood = 0
    for self_bd_num in self_gray[10]:
        self_bd = self_bd_num[0]
        # self blood gray pixel 80~98
        # 血量灰度值80~98
        if self_bd > 105 and self_bd < 109:
            self_blood += 2
        if self_bd == 105:
            self_blood += 1
    return self_blood
# def self_blood_count(self_gray):
#     self_blood = 0
#     for self_bd_num in self_gray[10]:
#         # self blood gray pixel 80~98
#         # 血量灰度值80~98
#         if self_bd_num > 80 and self_bd_num < 112:
#             self_blood += 2
#         if self_bd_num > 112 and self_bd_num < 205:
#             self_blood += 1
#     return self_blood
def take_action(action):
    print("action: ",action)
    if action == 0:     # n_choose
        pass
    elif action == 1:   # j
        directkeys.attack()
    elif action == 2:   # k
        directkeys.jump()
    elif action == 3:   # m
        directkeys.defense()
    elif action == 4:   # r
        directkeys.dodge()
    elif action == 5:   # j+k
        directkeys.right()
    elif action == 6:   # j+m
        directkeys.left()
    elif action == 7:   # j+r
        directkeys.people()
    elif action == 8:   # k+m
        directkeys.jump()
    elif action == 9:   # k+r
        directkeys.farattack()
    
def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, stop, emergence_break):
    # get action reward
    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    if next_self_blood < 3:     # self dead
        if emergence_break < 2:
            reward = -10
            done = 1
            stop = 0
            emergence_break += 1
            return reward, done, stop, emergence_break
        else:
            reward = -10
            done = 1
            stop = 0
            emergence_break = 100
            return reward, done, stop, emergence_break
    elif next_boss_blood - boss_blood > 15:   #boss dead
        if emergence_break < 2:
            reward = 20
            done = 0
            stop = 0
            emergence_break += 1
            return reward, done, stop, emergence_break
        else:
            reward = 20
            done = 0
            stop = 0
            emergence_break = 100
            return reward, done, stop, emergence_break
    else:
        self_blood_reward = 0
        boss_blood_reward = 0
        # print(next_self_blood - self_blood)
        # print(next_boss_blood - boss_blood)
        if next_self_blood - self_blood < -7:
            if stop == 0:
                self_blood_reward = -6
                stop = 1
                # 防止连续取帧时一直计算掉血
        else:
            stop = 0
        if next_boss_blood - boss_blood <= -3:
            boss_blood_reward = 4
        # print("self_blood_reward:    ",self_blood_reward)
        # print("boss_blood_reward:    ",boss_blood_reward)
        reward = self_blood_reward + boss_blood_reward
        done = 0
        emergence_break = 0
        return reward, done, stop, emergence_break
    
x, y, width, height = 80, 130, 260, 27
x2,y2,width,height = 370, 130, 260,27
x3,y3,width2,height2 = 60, 125, 600,500
if __name__ == "__main__":
    agent = DQN(HEIGHT, WIDTH,action_size, DQN_model_path, DQN_log_path)
    print("start")
    for episode in range(EPISODES):
        screen_gray = cv2.cvtColor(grab_screen(x3,y3,width2,height2),cv2.COLOR_BGR2GRAY)
        # blood_window_gray = cv2.cvtColor(grab_screen(blood_window),cv2.COLOR_BGR2GRAY)
        state = cv2.resize(screen_gray,(WIDTH,HEIGHT))
        my_blood_window_gray = grab_screen(x,y,width,height)
        enemy_blood_window_gray = grab_screen(x2,y2,width,height)
        boss_blood = 200 - self_blood_count(enemy_blood_window_gray)
        self_blood = self_blood_count(my_blood_window_gray)
        target_step = 0
        # used to update target Q network
        done = 0
        total_reward = 0
        stop = 0    
        emergence_break = 0
        # 用于防止连续帧重复计算reward
        last_time = time.time()
        state = torch.from_numpy(state).to(dtype=torch.float).to("mps")
        while True:
            state = state.reshape(-1,1,HEIGHT,WIDTH)[0]
            # state = state.unsqueeze(0)
            # print("state_shape",state.shape)
            print('loop took {} seconds'.format(time.time()-last_time))
            target_step += 1
            action = agent.choose_action(state)
            # print("choose_action: ",action)
            take_action(action)
            screen_gray = cv2.cvtColor(grab_screen(x3,y3,width,height),cv2.COLOR_RGB2GRAY)
            my_blood_window_gray = grab_screen(x,y,width,height)
            if my_blood_window_gray[10][30][0] == 0:
                  cv2.waitKey(100)
                  my_blood_window_gray = grab_screen(x,y,width,height)
            enemy_blood_window_gray = grab_screen(x2,y2,width,height)
            next_state = cv2.resize(screen_gray,(WIDTH,HEIGHT))
            next_state = np.array(next_state).reshape(-1,1,HEIGHT,WIDTH)[0]
            next_boss_blood = 200 - self_blood_count(enemy_blood_window_gray)
            next_self_blood = self_blood_count(my_blood_window_gray)
            print("next_boss_blood: ",next_boss_blood)
            print("next_self_blood: ",next_self_blood)
            reward, done, stop, emergence_break = action_judge(boss_blood, next_boss_blood,
                                                               self_blood, next_self_blood,
                                                               stop, emergence_break)
            if emergence_break == 100:
                # emergence break , save model and paused
                # 遇到紧急情况，保存数据，并且暂停
                print("emergence_break")
                agent.save_model()
                paused = True
            agent.store_data(state, action, reward,next_state, done)
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                num_step += 1
                # save loss graph
                # print('train')
                agent.train()
            if target_step % UPDATE_STEP == 0:
                agent.update_target()
            station = next_state
            self_blood = next_self_blood
            boss_blood = next_boss_blood
            total_reward += reward
            if done == 1:
                # print("done")
                break
            if episode % 10 == 0:
                agent.save_model()
            # save model
        print('episode: ', episode, 'Evaluation Average Reward:', total_reward/target_step)
        restart()

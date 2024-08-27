import torch
import torch.nn as nn
import random
from collections import deque
import cv2
from screenshot_utils import grab_game_screen, g_width, g_height
import keyboardandmouse as directkeys
from self_blood_utils import grab_screen_and_cal_AIplayer_blood
from boss_blood_utils import grab_screen_and_cal_boss_blood
from datetime import datetime
import os

# 设置超参数
REPLAY_SIZE = 2000
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
BATCH_SIZE = 16
GAMMA = 0.9
ACTION_SIZE = directkeys.action_size
MAX_EPISODES = 10000
SAVE_INTERVAL = 1000  # 模型保持的episodes间隔
UPDATE_STEP = 50  # target 网络更新频率
AIvsBoss_K = 1  # 协调更关注Boss血量还是自身血量
RECOVER_RATE = 5  # 协调完美闪避的重要度
MIN_WHITE_POINTS = 10  # 判定boss&AIplayer角色死亡
GAME_WIN_REWARD = 10000  # 胜利和失败的巨大惩罚

# 定义神经网络
class NET(nn.Module):
    def __init__(self, observation_height, observation_width, action_space):
        super(NET, self).__init__()
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        self.relu = nn.ReLU()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=[5,5], stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=[5,5], stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(int((self.state_w/4) * (self.state_h/4) * 64), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.action_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(-1, int((self.state_w/4) * (self.state_h/4) * 64))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN(object):
    def __init__(self, observation_height, observation_width, action_space, model_save_path, boss_name):
        self.model_save_path = model_save_path
        self.boss_name = boss_name
        self.target_net = NET(observation_height, observation_width, action_space)
        self.eval_net = NET(observation_height, observation_width, action_space)
        self.replay_buffer = deque(maxlen=REPLAY_SIZE)
        self.epsilon = INITIAL_EPSILON
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.action_dim = action_space

    def choose_action(self, state):
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            Q_value = self.eval_net(state)
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return torch.argmax(Q_value).item()

    def store_data(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = torch.stack([data[0] for data in minibatch])
        action_batch = torch.tensor([data[1] for data in minibatch])
        reward_batch = torch.tensor([data[2] for data in minibatch])
        next_state_batch = torch.stack([data[3] for data in minibatch])

        Q_value_batch = self.target_net(next_state_batch).detach()
        y_batch = reward_batch + GAMMA * torch.max(Q_value_batch, dim=1)[0] * (1 - torch.tensor([data[4] for data in minibatch]))

        Q_eval = self.eval_net(state_batch)
        Q_action = Q_eval.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        loss = self.loss(Q_action, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, prefix_=""):
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        boss_save_path = os.path.join(self.model_save_path,self.boss_name)
        if not os.path.exists(boss_save_path):
            os.mkdir(boss_save_path)
        file_name = f"{prefix_}{datetime.now().strftime('%Y_%m%d_%H%M')}.pth"
        model_path = os.path.join(boss_save_path, file_name)
        torch.save(self.target_net.state_dict(), model_path)

    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

def get_bloods_count():
    boss_blood, _ = grab_screen_and_cal_boss_blood()
    AIplayer_blood, _ = grab_screen_and_cal_AIplayer_blood()
    return boss_blood, AIplayer_blood

def check_done_condition():
    boss_blood, AIplayer_blood = get_bloods_count()
    if boss_blood < MIN_WHITE_POINTS:
        print("AI player win!")
        return 1, True
    if boss_blood < MIN_WHITE_POINTS:
        # print("AI player defeat!")
        print("Boss win!")
        return -1, True
    return None, False

# 游戏的训练流程
def train_game():
    model_save_path = "model_wukong"
    Boss_name = "huxianfeng"  # tiger

    # 截屏获取三个截图的坐标：game_screen, boss_blood, AIplayer_blood
    grab_game_screen()
    grab_screen_and_cal_AIplayer_blood()
    grab_screen_and_cal_boss_blood()
    print('monitors have been set!')

    print('construct DQN network...')
    WIDTH = g_width
    HEIGHT = g_height
    agent = DQN(HEIGHT, WIDTH, ACTION_SIZE, model_save_path,Boss_name)
    print('done.')

    last_boss_blood, last_AIplayer_blood = 0, 0
    first_flag = True
    print('training start...')
    for episode in range(MAX_EPISODES):
        screen_gray = cv2.cvtColor(grab_game_screen(), cv2.COLOR_BGR2GRAY)
        state = torch.from_numpy(cv2.resize(screen_gray, (WIDTH, HEIGHT))).float().unsqueeze(0)
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            directkeys.take_action(action)

            next_screen_gray = cv2.cvtColor(grab_game_screen(), cv2.COLOR_BGR2GRAY)
            next_state = torch.from_numpy(cv2.resize(next_screen_gray, (WIDTH, HEIGHT))).float().unsqueeze(0)

            boss_blood, AIplayer_blood = get_bloods_count()  # 使用之前的血量计算逻辑
            if first_flag:
                reward = 0
                last_boss_blood, last_AIplayer_blood = boss_blood, AIplayer_blood
            else:
                if AIplayer_blood > last_AIplayer_blood:
                    reward = RECOVER_RATE * AIvsBoss_K * (AIplayer_blood - last_AIplayer_blood) + max((last_boss_blood - boss_blood), 0)
                else:
                    reward = AIvsBoss_K * (AIplayer_blood - last_AIplayer_blood) + max((last_boss_blood - boss_blood), 0)

            is_win, done = check_done_condition()  # 判断是否结束
            if done:
                reward += is_win * GAME_WIN_REWARD

            agent.store_data(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

        if episode % UPDATE_STEP == 0:
            agent.update_target()

        if episode % SAVE_INTERVAL == 0 and episode != 0:
            agent.save_model()

        print(f'Episode {episode}, Total Reward: {total_reward}')
    print('training over.')
    agent.save_model('final_')
    print(f'final model saved.')


if __name__ == "__main__":
    # 启动训练
    train_game()

import time

import torch
import torch.nn as nn
import random
from collections import deque
import cv2
from screenshot_utils import grab_game_screen
import keyboardandmouse as directkeys
from self_blood_utils import grab_screen_and_cal_AIplayer_blood
from boss_blood_utils import grab_screen_and_cal_boss_blood
from datetime import datetime
import os
import pyautogui
import numpy as np

# 设置超参数
REPLAY_SIZE = 5000
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
BATCH_SIZE = 16
GAMMA = 0.9
ACTION_SIZE = directkeys.action_size
MAX_EPISODES = 2000
SAVE_INTERVAL = 100  # 模型保存的episodes间隔
UPDATE_STEP = 50  # target 网络更新频率
AIvsBoss_K = 10  # 协调更关注Boss血量还是自身血量
RECOVER_RATE = 2  # 协调完美闪避的重要度
MIN_WHITE_POINTS_RATE = 0.1  # 判定boss&AIplayer角色死亡,相比初始的血量白点的比例
DRINK_ADD_WHITE_POINTS_RATE = 0.15  # 喝口药相比初始的血量白点的比例
GAME_WIN_REWARD = 1000  # 胜利和失败的巨大惩罚
WIDTH = 64  # 输入net的图像长度
HEIGHT = 64  # 输入net的图像高度
TRANSFORM_PROB = 0.8
RECFLU_AI, RECFLU_BOSS = 50, 50  # Recognition fluctuation

# 控制喝酒的逻辑
DRINK_FLAG = False

# 定义图像处理网络
class NET(nn.Module):
    def __init__(self, observation_height, observation_width, action_space):
        super(NET, self).__init__()
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        self.relu = nn.ReLU()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=[5,5], stride=1, padding=2),  # , padding='same'
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=[5,5], stride=1, padding=2),  # , padding='same'
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
        self.replay_buffer.append([state, action, reward, next_state, done])

    def update_latest_several_actions_rewards(self, reward):
        self.replay_buffer[-1][2] = self.replay_buffer[-1][2] + reward # * 0.6  # * 0.25
        self.replay_buffer[-2][2] = self.replay_buffer[-2][2] + reward # * 0.8  # * 0.5
        self.replay_buffer[-3][2] = self.replay_buffer[-3][2] + reward
        # self.replay_buffer[-4][2] = self.replay_buffer[-4][2] + reward
        # self.replay_buffer[-5][2] = self.replay_buffer[-5][2] + reward

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = torch.stack([data[0] for data in minibatch])
        action_batch = torch.tensor([data[1] for data in minibatch])  # , dtype=torch.float32
        reward_batch = torch.tensor([data[2] for data in minibatch], dtype=torch.float32)
        next_state_batch = torch.stack([data[3] for data in minibatch])

        Q_value_batch = self.target_net(next_state_batch).detach()
        y_batch = reward_batch + GAMMA * torch.max(Q_value_batch, dim=1)[0] * (1 - torch.tensor([int(data[4]) for data in minibatch]))

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
    boss_blood, _, _ = grab_screen_and_cal_boss_blood()
    AIplayer_blood, _, _ = grab_screen_and_cal_AIplayer_blood()
    return boss_blood, AIplayer_blood

def check_done_condition():
    global initial_boss_blood, initial_AIplayer_blood
    boss_blood, AIplayer_blood = get_bloods_count()
    if AIplayer_blood < MIN_WHITE_POINTS_RATE * initial_AIplayer_blood and boss_blood < MIN_WHITE_POINTS_RATE * initial_boss_blood:
        print('all bloods are few.')
        print("Boss win!")
        return -1, True
    if AIplayer_blood < MIN_WHITE_POINTS_RATE * initial_AIplayer_blood:  # MIN_WHITE_POINTS
        # print("AI player defeat!")
        print(f"AIplayer_blood: {AIplayer_blood} < {MIN_WHITE_POINTS_RATE * initial_AIplayer_blood}")
        print("Boss win!")
        return -1, True
    if boss_blood < MIN_WHITE_POINTS_RATE * initial_boss_blood:  # MIN_WHITE_POINTS
        print(f"boss_blood: {boss_blood} < {MIN_WHITE_POINTS_RATE * initial_boss_blood}")
        print("AI player win!")
        return 1, True
    return None, False

def check_done_condition_new(initial_boss_blood,boss_blood,initial_AIplayer_blood,AIplayer_blood):
    if AIplayer_blood < MIN_WHITE_POINTS_RATE * initial_AIplayer_blood and boss_blood < MIN_WHITE_POINTS_RATE * initial_boss_blood:  # MIN_WHITE_POINTS
        # print("AI player defeat!")
        # print(f"AIplayer_blood: {AIplayer_blood} < {MIN_WHITE_POINTS_RATE * initial_AIplayer_blood}")
        print('all bloods are few.')
        print("Boss win!")
        return -1, True
    if AIplayer_blood < MIN_WHITE_POINTS_RATE * initial_AIplayer_blood:  # MIN_WHITE_POINTS
        # print("AI player defeat!")
        print(f"AI　player　blood: {AIplayer_blood} < {MIN_WHITE_POINTS_RATE * initial_AIplayer_blood}")
        print("Boss win!")
        return -1, True
    if boss_blood < MIN_WHITE_POINTS_RATE * initial_boss_blood:  # MIN_WHITE_POINTS
        print(f"Boss　blood: {boss_blood} < {MIN_WHITE_POINTS_RATE * initial_boss_blood}")
        print("AI　player win!")
        return 1, True
    return None, False


# 对灰度图像的数据增强不需要太多
"""
if np.random.rand() > TRANSFORM_PROB:
    screen_gray = cv2.applyColorMap(screen_gray, cv2.COLORMAP_JET)
if np.random.rand() > TRANSFORM_PROB:
    noise = np.random.normal(0, 10, screen_gray.shape).astype(np.uint8)
    screen_gray = cv2.add(screen_gray, noise)
if np.random.rand() > TRANSFORM_PROB:
    brightness_factor = np.random.uniform(0.8, 1.2)
    screen_gray = cv2.convertScaleAbs(screen_gray, alpha=brightness_factor, beta=0)
if np.random.rand() > TRANSFORM_PROB:
    contrast_factor = np.random.uniform(0.8, 1.2)
    screen_gray = cv2.convertScaleAbs(screen_gray, alpha=contrast_factor, beta=0)
#TODO
# if np.random.rand() > TRANSFORM_PROB:
#     pass
"""

"""
h, w = image_gray.shape
center_x, center_y = w // 2, h // 2
scale_factor = np.random.uniform(0.95, 1.05)
angle = np.random.uniform(-5, 5)
M_rotate_scale = cv2.getRotationMatrix2D((center_x, center_y), angle, scale_factor)
image_gray = cv2.warpAffine(image_gray, M_rotate_scale, (w, h))
"""
def gray_image_augmentation(image_gray):
    if np.random.rand() > TRANSFORM_PROB:  # 仿射变换，调整角度  # XXX 以及长宽
        h, w = image_gray.shape
        center_x, center_y = w // 2, h // 2
        angle = np.random.uniform(-5, 5)
        M_rotate = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        image_gray = cv2.warpAffine(image_gray, M_rotate, (w, h))
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0) if np.random.rand() > TRANSFORM_PROB else image_gray  # cv2.blur均值模糊
    if np.random.rand() > TRANSFORM_PROB:
        noise = np.random.normal(0, 10, image_gray.shape).astype(np.uint8)
        image_gray = cv2.add(image_gray, noise)
    image_gray = cv2.convertScaleAbs(image_gray, alpha=np.random.uniform(0.8, 1.2), beta=0) if np.random.rand() > TRANSFORM_PROB else image_gray
    image_gray = cv2.convertScaleAbs(image_gray, alpha=1, beta=np.random.randint(-50, 50)) if np.random.rand() > TRANSFORM_PROB else image_gray
    #TODO image_gray = ...
    return image_gray

# 游戏的训练流程
def train_game():
    global DRINK_FLAG, initial_AIplayer_blood  # initial_boss_blood,
    model_save_path = "model_wukong"
    Boss_name = "xiaoguai1"  # tiger yanhu huxianfeng xiaoguai1

    print('construct DQN network...')
    WUKONGagent = DQN(HEIGHT, WIDTH, ACTION_SIZE, model_save_path,Boss_name)
    print('done.')

    print('training start...')
    for episode in range(MAX_EPISODES):
        print(f'Episode [{episode}] start.')
        last_boss_blood, last_AIplayer_blood = 0, 0
        first_flag = True
        screen_gray = cv2.cvtColor(grab_game_screen(), cv2.COLOR_BGR2GRAY)
        screen_gray = gray_image_augmentation(screen_gray)
        state = torch.from_numpy(cv2.resize(screen_gray, (WIDTH, HEIGHT))).float().unsqueeze(0)
        done = False
        total_reward = 0
        last_drink_time = time.time()
        last_action_choose = time.time()
        while not done:
            action = WUKONGagent.choose_action(state.unsqueeze(0))
            directkeys.take_action(action)
            action_choose = time.time()
            print('action interval:',action_choose-last_action_choose)
            last_action_choose = action_choose
            next_screen_gray = cv2.cvtColor(grab_game_screen(), cv2.COLOR_BGR2GRAY)
            next_screen_gray = gray_image_augmentation(next_screen_gray)
            next_state = torch.from_numpy(cv2.resize(next_screen_gray, (WIDTH, HEIGHT))).float().unsqueeze(0)
            boss_blood, AIplayer_blood = get_bloods_count()  # 使用之前的血量计算逻辑
            #TODO
            # if AIplayer_blood < initial_AIplayer_blood / 3:  # 自动喝葫芦
            #     if time.time()-last_drink_time > 3:
            #         pyautogui.press(['r', 'r'])  # 键盘上点击三~五次喝药，避免其他动作打断 , 'r', 'r', 'r'
            #         # pyautogui.press('r')
            #         print('脚本按了 r 喝药.')
            #         DRINK_FLAG = True
            #         last_drink_time = time.time()

            if first_flag:
                reward = 0
                first_flag = False
            else:
                # 计算攻击boss，血量降低的奖励
                Boss_blood_reward = max((last_boss_blood - boss_blood), 0) if 0.5* initial_boss_blood > abs((last_boss_blood - boss_blood)) > RECFLU_BOSS else 0
                # 计算AIplayer的血量上升/降低的奖励/惩罚
                if AIplayer_blood - last_AIplayer_blood > RECFLU_AI:  # AIplayer_blood > last_AIplayer_blood and
                    if DRINK_FLAG and ((AIplayer_blood-last_AIplayer_blood) > DRINK_ADD_WHITE_POINTS_RATE * initial_AIplayer_blood):  # DRINK_ADD_WHITE_POINTS
                        reward = Boss_blood_reward
                        DRINK_FLAG = False
                    elif AIplayer_blood - last_AIplayer_blood > 0.5 * initial_AIplayer_blood:  # 异常情况，不看自身的血量奖励了
                        reward = Boss_blood_reward
                    else:
                        reward = RECOVER_RATE * AIvsBoss_K * (AIplayer_blood - last_AIplayer_blood) + Boss_blood_reward
                elif last_AIplayer_blood - AIplayer_blood > RECFLU_AI:
                    reward = AIvsBoss_K * (AIplayer_blood - last_AIplayer_blood) + Boss_blood_reward
                else:
                    # AIvsBoss_K * (AIplayer_blood - last_AIplayer_blood) +
                    # 这是AIplayer的识别波动，AIplayer的血量不予考虑
                    reward = Boss_blood_reward
                if reward != 0:
                    print(f'Boss reward: {Boss_blood_reward}, AI reward: {AIplayer_blood - last_AIplayer_blood}, reward sum: {reward}')
            # print(f'Boss 初始血量{initial_boss_blood}, 上次血量:{last_boss_blood}, 现在血量{boss_blood}, 变化 {boss_blood-last_boss_blood}.')
            # print(f'AI player 初始血量 {initial_AIplayer_blood}, 上次血量 {last_AIplayer_blood}, 现在血量 {AIplayer_blood}, 变化 {AIplayer_blood-last_AIplayer_blood}.')
            last_boss_blood, last_AIplayer_blood = boss_blood, AIplayer_blood

            is_win, done = check_done_condition_new(initial_boss_blood,boss_blood,initial_AIplayer_blood,AIplayer_blood)  # 判断是否结束
            # if done:
            #     reward += is_win * GAME_WIN_REWARD
            # 防止某一帧出不同步的bug
            if done:
                for _ in range(5):
                    time.sleep(0.25)
                    is_win, done_ = check_done_condition()
                    if not done_:
                        done = False
                        break

            WUKONGagent.store_data(state, action, reward, next_state, done)
            if len(WUKONGagent.replay_buffer) > 10:
                WUKONGagent.update_latest_several_actions_rewards(reward)
            WUKONGagent.train()

            state = next_state
            total_reward += reward

        if episode % UPDATE_STEP == 0:
            WUKONGagent.update_target()

        if episode % SAVE_INTERVAL == 0 and episode != 0:
            WUKONGagent.save_model()

        print(f'Episode {episode}, Total Reward: {total_reward}')

        # 通过mod进入boss区域
        b_blood, A_blood = get_bloods_count()
        # b_blood < 0.5 * initial_boss_blood and A_blood < 0.5 * initial_AIplayer_blood
        while b_blood < 0.5 * initial_boss_blood:
            print('AI 血量:', A_blood, "v.s.", '满足重置条件最低血量', 0.5 * initial_AIplayer_blood, '初始总血量',
                  initial_AIplayer_blood)
            print('Boss 血量:', b_blood, "v.s.", '满足重置条件最低血量', 0.5 * initial_boss_blood, '初始总血量', initial_boss_blood)
            time.sleep(3)
            pyautogui.press('l')
            print('脚本按了 l 复位.')
            b_blood, A_blood = get_bloods_count()

    print('training over.')
    WUKONGagent.save_model('final_')
    print(f'final model saved.')

DEBUG = True
if __name__ == "__main__":
    if DEBUG:
        game_image = grab_game_screen(0, 34, 1276, 714)  # 4 31 1267 717 # 200, 200, 100, 100
        # cv2.imshow(f"Captured Game Image", game_image)
        cv2.imwrite('train_images/game_image.jpg', game_image)
        initial_AIplayer_blood, AI_white_pixel_mask, AI_blood_image = grab_screen_and_cal_AIplayer_blood(126, 675, 267, 29)  # 126 670 157 29 # 400, 400, 100, 100
        black_white_image = np.zeros_like(AI_blood_image[:, :, 0])
        black_white_image[AI_white_pixel_mask] = 255
        cv2.imwrite('train_images/black_white_AIplayer_blood.jpg', black_white_image)

        # cv2.   imshow(f"Captured AI blood Image", AI_blood_image)
        cv2.imwrite('train_images/AI_blood_image.jpg', AI_blood_image)
        initial_boss_blood, Boss_white_pixel_mask, Boss_blood_image = grab_screen_and_cal_boss_blood(475, 625, 343, 41)  # 600, 600, 100, 100
        # cv2.imshow(f"Captured Boss blood Image", Boss_blood_image)
        cv2.imwrite('train_images/Boss_blood_image.jpg', Boss_blood_image)
        black_white_image = np.zeros_like(Boss_blood_image[:, :, 0])
        black_white_image[Boss_white_pixel_mask] = 255
        cv2.imwrite('train_images/black_white_boss_blood.jpg', black_white_image)

        # initial_AIplayer_blood, initial_boss_blood = 100, 100
    else:
        # 截屏获取三个截图的坐标：game_screen, boss_blood, AIplayer_blood
        print('monitors setting...')
        print('设置游戏画面监视.')
        grab_game_screen()
        time.sleep(1)
        print('设置AI player血量监视.')
        initial_AIplayer_blood, _, _ = grab_screen_and_cal_AIplayer_blood()
        time.sleep(1)
        print('设置boss血量监视.')
        initial_boss_blood, _, _ = grab_screen_and_cal_boss_blood()
        # print('monitors have been set.')
        print('done.')

    # 启动训练
    train_game()

import torch
import cv2
from training import NET  # 假设training.py文件中的NET类已经定义并存储在同一目录下
from screenshot_utils import grab_game_screen
import keyboardandmouse as directkeys
import os

# 定义测试函数
def test_game(model_path, boss_name, episodes=10):
    WIDTH, HEIGHT = 160, 90  # 根据实际情况设置
    ACTION_SIZE = 10  # 根据训练中的ACTION_SIZE设置

    # 实例化网络
    model = NET(HEIGHT, WIDTH, ACTION_SIZE)
    model_load_path = os.path.join(model_path, boss_name)
    checkpoints = sorted(os.listdir(model_load_path), reverse=True)  # 加载最新的checkpoint
    if len(checkpoints) == 0:
        print("No checkpoint found!")
        return

        # 列出并过滤出具有 .pth 或 .pt 后缀的文件
    checkpoints = sorted(
        [f for f in os.listdir(model_load_path) if f.endswith('.pth') or f.endswith('.pt')],
        reverse=True
    )
    if len(checkpoints) == 0:
        print("No checkpoint found!")
        return
    print("Available checkpoints:")
    for idx, checkpoint in enumerate(checkpoints):
        print(f"{idx}: {checkpoint}")
    chosen_index = int(input("Enter the index of the checkpoint to load: "))
    if chosen_index < 0 or chosen_index >= len(checkpoints):
        print("Invalid index selected!")
        return

    # 加载用户选择的checkpoint
    checkpoint = os.path.join(model_load_path, checkpoints[chosen_index])
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    for episode in range(episodes):
        screen_gray = cv2.cvtColor(grab_game_screen(), cv2.COLOR_BGR2GRAY)
        state = torch.from_numpy(cv2.resize(screen_gray, (WIDTH, HEIGHT))).float().unsqueeze(0).unsqueeze(0)  # 增加一个维度以匹配模型输入
        done = False

        while not done:
            with torch.no_grad():
                Q_value = model(state)
                action = torch.argmax(Q_value).item()

            directkeys.take_action(action)

            next_screen_gray = cv2.cvtColor(grab_game_screen(), cv2.COLOR_BGR2GRAY)
            next_state = torch.from_numpy(cv2.resize(next_screen_gray, (WIDTH, HEIGHT))).float().unsqueeze(0).unsqueeze(0)

            state = next_state
            # 如果有终止条件，可以设置done=True以结束循环

        print(f'Episode {episode}, Action Taken: {action}')

if __name__ == "__main__":
    model_save_path = "model_wukong"
    boss_name = "huxianfeng"  # tiger

    # 开始测试
    test_game(model_save_path, boss_name, episodes=10)

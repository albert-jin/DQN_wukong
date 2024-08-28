import pyautogui
import time
import threading

# pyautogui.hotkey('ctrl', 'a')  # 全选
USE_HOTKEY = False
USE_DURATION = False

PASS = True

def key_move_thr(key,duration):
    pyautogui.keyDown(key)
    pyautogui.keyDown('shift')
    time.sleep(duration)
    pyautogui.keyUp('shift')
    pyautogui.keyUp(key)

"""默认所有移动动作 都加了 sprint 空格奔跑"""
def move_forward(duration=1.2):
    if USE_HOTKEY:
        pyautogui.hotkey('w', 'shift', interval=duration)
    else:
        threading.Thread(target=key_move_thr, args=('w',duration)).start()

def move_backward(duration=1.2):
    if USE_HOTKEY:
        pyautogui.hotkey('s', 'shift', interval=duration)
    else:
        threading.Thread(target=key_move_thr, args=('s', duration)).start()

def move_left(duration=1.2):
    if USE_HOTKEY:
        pyautogui.hotkey('a', 'shift', interval=duration)
    else:
        threading.Thread(target=key_move_thr, args=('a', duration)).start()

def move_right(duration=1.2):
    if USE_HOTKEY:
        pyautogui.hotkey('d', 'shift', interval=duration)
    else:
        threading.Thread(target=key_move_thr, args=('d', duration)).start()

def dodge():
    pyautogui.press('space')  # 按下空格键进行躲避

def jump():
    pyautogui.press('ctrl')  # 按下ctrl键进行跳跃

def light_attack():
    pyautogui.click(button='left')  # 左键点击进行轻攻击

def heavy_attack_0():
    # 右键点击进行重攻击(无蓄力)
    pyautogui.click(button='right')

def mouse_heavy_attack_thr(button, duration):
    pyautogui.mouseDown(button=button)  # 按下右键
    time.sleep(duration)  # 持续指定时间
    pyautogui.mouseUp(button=button)  # 松开右键

def heavy_attack_1():
    if PASS:
        return
    # 右键点击进行重攻击(蓄力至一颗豆)
    if USE_DURATION:
        pyautogui.mouseDown(button='right', duration=1)  # 按下右键
    else:
        threading.Thread(target=mouse_heavy_attack_thr, args=('right', 1)).start()

def heavy_attack_2():
    if PASS:
        return
    # 右键点击进行重攻击(蓄力至两颗豆)
    if USE_DURATION:
        pyautogui.mouseDown(button='right', duration=2)  # 按下右键
    else:
        threading.Thread(target=mouse_heavy_attack_thr, args=('right', 2)).start()


def heavy_attack_3():
    if PASS:
        return
    # 右键点击进行重攻击(蓄力至三颗豆)
    if USE_DURATION:
        pyautogui.mouseDown(button='right', duration=3)  # 按下右键
    else:
        threading.Thread(target=mouse_heavy_attack_thr, args=('right', 3)).start()

def do_nothing():
    # 什么也不做
    pass
    return

# move_forward,move_backward,move_left,move_right,jump,heavy_attack_1,heavy_attack_2,heavy_attack_3,
action_set = [dodge,light_attack,light_attack,do_nothing]  # ,heavy_attack_0 重击没用
action_size = len(action_set)

def take_action(action_idx):
    action_set[action_idx]()
    print("do action:", action_set[action_idx].__name__)

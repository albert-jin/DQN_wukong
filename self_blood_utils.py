import time

import cv2
import pyautogui
import numpy as np
import keyboard

g_AIplayerb_x, g_AIplayerb_y, g_AIplayerb_width, g_AIplayerb_height, g_AIplayerb_flag = None, None, None, None, False


def select_AIplayer_blood_region():
    """
    让用户通过按两次 'J' 键选择屏幕区域，并返回选择区域的坐标和尺寸。

    返回:
        tuple: 选定区域的左上角坐标 (x, y) 和尺寸 (width, height)。
    """
    print("请将光标移动到选定区域的左上角，按下 'capslock' 或 'f7' 键。")
    while True:
        if keyboard.is_pressed('capslock') or keyboard.is_pressed('f7'):
            start_x, start_y = pyautogui.position()
            print(f"左上角位置已记录: ({start_x}, {start_y})")
            break
    time.sleep(1)
    print("请将光标移动到选定区域的右下角，按下 'capslock' 或 'f7' 键。")
    while True:
        if keyboard.is_pressed('capslock') or keyboard.is_pressed('f7'):
            end_x, end_y = pyautogui.position()
            print(f"右下角位置已记录: ({end_x}, {end_y})")
            break

    # 计算区域的宽度和高度
    width = end_x - start_x
    height = end_y - start_y

    print(f"你选择的区域是：左上角 ({start_x}, {start_y}) 宽度: {width}, 高度: {height}")
    return start_x, start_y, width, height


def grab_screen_and_cal_AIplayer_blood(x=None, y=None, width=None, height=None):
    """
    截取屏幕上指定AIplayer血量图像,并统计血量多少。

    参数:
        x (int): 截图区域左上角的x坐标。
        y (int): 截图区域左上角的y坐标。
        width (int): 截图区域的宽度。
        height (int): 截图区域的高度。

    返回:
        np.ndarray: 截取的图像，格式为NumPy数组（BGR颜色空间）。
    """
    global g_AIplayerb_flag, g_AIplayerb_x, g_AIplayerb_y, g_AIplayerb_width, g_AIplayerb_height
    if x is None and y is None and width is None and height is None and g_AIplayerb_flag is False:
        x, y, width, height = select_AIplayer_blood_region()
        g_AIplayerb_x, g_AIplayerb_y, g_AIplayerb_width, g_AIplayerb_height = x, y, width, height
        print('g_AIplayerb_x, g_AIplayerb_y, g_AIplayerb_width, g_AIplayerb_height:', g_AIplayerb_x, g_AIplayerb_y,
              g_AIplayerb_width, g_AIplayerb_height)
        g_AIplayerb_flag = True
        # 截取屏幕区域的图像
        screenshot = pyautogui.screenshot(region=(g_AIplayerb_x, g_AIplayerb_y, g_AIplayerb_width, g_AIplayerb_height))
    elif x is None and y is None and width is None and height is None:
        # 截取屏幕区域的图像
        screenshot = pyautogui.screenshot(region=(g_AIplayerb_x, g_AIplayerb_y, g_AIplayerb_width, g_AIplayerb_height))
    else:
        # 截取屏幕区域的图像
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        g_AIplayerb_x, g_AIplayerb_y, g_AIplayerb_width, g_AIplayerb_height = x, y, width, height
        g_AIplayerb_flag = True

    # 将 PIL 图像转换为 NumPy 数组
    frame_ori = np.array(screenshot)

    # 将图像从 RGB 转换为 BGR 颜色空间，以便 OpenCV 处理
    frame = cv2.cvtColor(frame_ori, cv2.COLOR_RGB2BGR)  # frame_ori if frame_ori else

    """
        计算图像中白色像素的数量。白色像素定义为RGB三原色的值都在200到255之间。
        参数: image (np.ndarray): 输入的图像，形状为 (height, width, 3)，表示RGB图像。
        返回: int: 图像中白色像素的数量。
    """
    # 条件为所有像素点的RGB值均在200到255之间
    # white_pixel_mask = (frame[:, :, 0] + frame[:, :, 1] + frame[:, :, 2])/3 > 175  # (frame[:, :, 0] > 125) & (frame[:, :, 1] > 125) & (frame[:, :, 2] > 125)

    # gray_frame = np.mean(frame, axis=2)
    # white_pixel_mask = gray_frame > 200

    mask1 = (frame[:, :, 0] > 125) & (frame[:, :, 1] > 125) & (frame[:, :, 2] > 175)
    """   
    red_green_diff = np.abs(frame[:, :, 0] - frame[:, :, 1])
    blue_green_diff = np.abs(frame[:, :, 2] - frame[:, :, 1])
    blue_red_diff = np.abs(frame[:, :, 2] - frame[:, :, 0])
    # mask2 = max(red_green_diff, blue_green_diff, blue_red_diff) <= 60
    max_diff = np.maximum(np.maximum(red_green_diff, blue_green_diff), blue_red_diff)
    mask2 = max_diff <= 50
    """
    white_pixel_mask = mask1  # & mask2  # final_mask
    """   
        (frame[:, :, 0] > 200) & (frame[:, :, 0] <= 255) & \
        (frame[:, :, 1] > 200) & (frame[:, :, 1] <= 255) & \
        (frame[:, :, 2] > 200) & (frame[:, :, 2] <= 255)
    """
    # 计算白色像素的总数
    white_pixel_count = np.sum(white_pixel_mask)
    return white_pixel_count, white_pixel_mask, frame


if __name__ == '__main__':
    for idx in range(5):
        white_pixel_count, white_pixel_mask, captured_image = grab_screen_and_cal_AIplayer_blood()
        cv2.imshow(f"Captured Image 0_{idx}", captured_image)
        print(f"Captured Image 0_{idx},", g_AIplayerb_x, g_AIplayerb_y, g_AIplayerb_width, g_AIplayerb_height)
        print(f'AIplayer blood: {white_pixel_count}.')
        cv2.waitKey(2000)  # 显示2秒
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()

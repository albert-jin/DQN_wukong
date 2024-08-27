import time

import cv2
import pyautogui
import numpy as np
import keyboard

g_x, g_y, g_width, g_height, g_flag = None, None, None, None, False


def select_screen_region():
    """
    让用户通过按两次 'J' 键选择屏幕区域，并返回选择区域的坐标和尺寸。

    返回:
        tuple: 选定区域的左上角坐标 (x, y) 和尺寸 (width, height)。
    """
    print("请将光标移动到选定区域的左上角，按下 '空格' 键。")
    while True:
        if keyboard.is_pressed('space'):
            start_x, start_y = pyautogui.position()
            print(f"左上角位置已记录: ({start_x}, {start_y})")
            break
    time.sleep(1)
    print("请将光标移动到选定区域的右下角，按下 '空格' 键。")
    while True:
        if keyboard.is_pressed('space'):
            end_x, end_y = pyautogui.position()
            print(f"右下角位置已记录: ({end_x}, {end_y})")
            break

    # 计算区域的宽度和高度
    width = end_x - start_x
    height = end_y - start_y

    print(f"你选择的区域是：左上角 ({start_x}, {start_y}) 宽度: {width}, 高度: {height}")
    return start_x, start_y, width, height


def grab_game_screen(x=None, y=None, width=None, height=None):
    """
    截取屏幕上指定区域的图像。

    参数:
        x (int): 截图区域左上角的x坐标。
        y (int): 截图区域左上角的y坐标。
        width (int): 截图区域的宽度。
        height (int): 截图区域的高度。

    返回:
        np.ndarray: 截取的图像，格式为NumPy数组（BGR颜色空间）。
    """
    global g_flag, g_x, g_y, g_width, g_height
    if x is None and y is None and width is None and height is None and g_flag is False:
        x, y, width, height = select_screen_region()
        g_x, g_y, g_width, g_height = x, y, width, height
        g_flag = True
        # 截取屏幕区域的图像
        screenshot = pyautogui.screenshot(region=(g_x, g_y, g_width, g_height))
    elif x is None and y is None and width is None and height is None:
        # 截取屏幕区域的图像
        screenshot = pyautogui.screenshot(region=(g_x, g_y, g_width, g_height))
    else:
        # 截取屏幕区域的图像
        screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # 将 PIL 图像转换为 NumPy 数组
    frame = np.array(screenshot)

    # 将图像从 RGB 转换为 BGR 颜色空间，以便 OpenCV 处理
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


if __name__ == '__main__':
    for idx in range(5):
        captured_image = grab_game_screen()
        cv2.imshow(f"Captured Image 0_{idx}", captured_image)
        print(f"Captured Image 0_{idx},", g_x, g_y, g_width, g_height)
        cv2.waitKey(2000)  # 显示2秒

    """  
        # 测试 function select_screen_region
        x, y, width, height = select_screen_region()
        captured_image_0 = grab_game_screen(x, y, width, height)
        cv2.imshow(f"Captured Image 0", captured_image_0)
        cv2.waitKey(2000)  # 显示2秒
    """

    """
        # 测试 function grab_game_screen
        # 定义两个截屏区域的坐标和尺寸
        region_1 = (75, 125, 260, 270)
        region_2 = (370, 125, 260, 270)
        # 捕获并显示第一个区域的图像
        captured_image_1 = grab_game_screen(*region_1)
        cv2.imshow("Captured Image 1", captured_image_1)
        cv2.waitKey(2000)  # 显示2秒
    
        # 捕获并显示第二个区域的图像
        captured_image_2 = grab_game_screen(*region_2)
        cv2.imshow("Captured Image 2", captured_image_2)
        cv2.waitKey(2000)  # 显示2秒
    """
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()

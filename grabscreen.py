import cv2
import pyautogui
import numpy as np

def grab_screen(x, y, width, height):
    # 截取屏幕上指定区域的图像
    screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # 将 PIL 图像转换为 NumPy 数组，以便 OpenCV 可以处理
    frame = np.array(screenshot)

    # OpenCV 默认使用 BGR 颜色空间，而 pyautogui 截图为 RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Captured Image", frame)
    # cv2.waitKey(5)
    return frame

if __name__ == '__main__':
# 指定区域的坐标和尺寸
    x, y, width, height = 75, 125, 260, 27
    x2,y2,width,height = 370, 125, 260,27
    # 捕获指定区域
    captured_image = grab_screen(x, y, width, height)
    captured_image = grab_screen(x2, y2, width, height)
    # 使用 OpenCV 显示图像
    cv2.imshow("Captured Image", captured_image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

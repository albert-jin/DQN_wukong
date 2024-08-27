# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:31:36 2020

@author: pang
"""
import directkeys
import time
import pyautogui
def restart():
    print("死,restart")
    time.sleep(12)
    pyautogui.press('Y')
    print("Y")
    time.sleep(0.2)
    pyautogui.press('J')
    time.sleep(0.5)
    pyautogui.press('J')
    time.sleep(10)
    directkeys.attack()
    print("开始新一轮")
  
if __name__ == "__main__":  
    # time.sleep(5)
    directkeys.attack()
    time.sleep(0.1)
    pyautogui.press('J')
    time.sleep(0.2)
    pyautogui.press('J')
    restart()
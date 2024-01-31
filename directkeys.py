# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:50 2020

@author: pang
"""

# import Quartz
# import time
# from Quartz.CoreGraphics import CGEventCreateKeyboardEvent, CGEventPost, kCGHIDEventTap
import pyautogui
# def key_down(key_code):
#     event = CGEventCreateKeyboardEvent(None, key_code, True)
#     CGEventPost(kCGHIDEventTap, event)

# def key_up(key_code):
#     event = CGEventCreateKeyboardEvent(None, key_code, False)
#     CGEventPost(kCGHIDEventTap, event)
# def press_key(key_code):
#     key_down(key_code)
#     time.sleep(0.05)
#     key_up(key_code)
# def left():
#     key_down(0x7B)
#     time.sleep(0.1)
#     key_up(0x7B)
#     time.sleep(0.2)
# def right():  
#     key_down(0x7C)
#     time.sleep(0.1)
#     key_up(0x7C)
#     time.sleep(0.2)
# def attack():
#     key_down(0x04)
#     time.sleep(0.1)
#     key_up(0x04)
#     time.sleep(0.2)
# def move():
#     key_down(0x0D)
#     time.sleep(0.1)
#     key_up(0x0D)
#     time.sleep(0.2)
# def jump():
#     key_down(0x0C)
#     time.sleep(0.1)
#     key_up(0x0C)
#     time.sleep(0.2)
# def defense():
#     key_down(0x0E)
#     time.sleep(0.1)
#     key_up(0x0E)
#     time.sleep(0.2)
# def dodge():
#     key_down(0x0F)
#     time.sleep(0.1)
#     key_up(0x0F)
#     time.sleep(0.2)

def right():
    pyautogui.press('D')
def left():
    pyautogui.press('A')
def defense():
    pyautogui.press('S')
def attack():
    pyautogui.press('J')
def jump():
    pyautogui.press('K')
def dodge():
    pyautogui.press('L')
def ultra():
    pyautogui.press('I')
def people():
    pyautogui.press('O')
def farattack():
    pyautogui.press('U')
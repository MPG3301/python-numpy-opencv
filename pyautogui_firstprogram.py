import pyautogui as pag
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
 
Tk().withdraw()
filename = askopenfilename()
print(filename)
 
timeDelay = int(input("If you want a delay, enter the number of seconds for the delay : ").split()[0])
 
if timeDelay < 1:
    timeDelay = 1
 
time.sleep(5)
 
f = open(filename, "r")
for word in f:
    time.sleep(timeDelay)
    pag.typewrite(word)
    pag.press("enter")

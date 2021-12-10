from tkinter import *
from tkinter import simpledialog
import numpy as np

def start(move_var, color):
	s = simpledialog.askstring("input string", "White or black")
	if s == 'w' or s == 'W' or s == 'white' or s == 'White':
		color = 'white'
		move_var = None
	elif s == 'b' or s == 'B' or s == 'black' or s == 'Black':
		m = simpledialog.askstring("input string", "First Move")
		color = 'black'
		move_var = m

def abort():
	quit()


def UI(var, color):
	root = Tk()
	root.geometry("300x300")

	Start_btn = Button(root, text='Strart Playing Chess', padx=50, command=start(var, color))
	Start_btn.pack()
	Abort = Button(root, text='Abort', padx=20, command=abort)
	Abort.pack()

	root.mainloop()
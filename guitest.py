from tkinter import *
from tkinter import messagebox
import use_model as um

top = Tk()
top.geometry("500x500")

def testing():
	str = E1.get()
	prediction = um.get_prediction(str)
	if prediction == 1:
		messagebox.showinfo("WARNING","The URL entered may be a Phishing URL")
	else:
		messagebox.showinfo("Info","The URL entered is benign")

L1 = Label(top,text = "URL",font=(None, 10))
L1.place(x = 65,y = 223)

E1 = Entry(top, width = 50,bd = 5)
E1.place(x = 100,y = 220)

B = Button(top,text = "Check",command = testing,width = 10)
B.place(x = 200,y = 250)

top.mainloop()

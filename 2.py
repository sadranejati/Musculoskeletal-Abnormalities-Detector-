from tkinter import *
from PIL import Image, ImageTk
import cv2

root = Tk()
root.geometry("1200x700")

label =Label(root)
label.grid(row=0, column=0)
cap= cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2image = None

def show_frames():
    global cv2image
    cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image = img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(20, show_frames)

def key_pressed(event):
    take_pic()

i = 0
def take_pic():
    global i
    cv2.imwrite(f"pics/{i}.jpg", cv2image)
    i += 1
    ###Rest of the photo saving script

show_frames()
root.bind("<Key>", key_pressed)
root.mainloop()
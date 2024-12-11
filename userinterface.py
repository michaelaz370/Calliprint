"""
ajout progress bar
"""

import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
import ttkbootstrap as tb
from ttkbootstrap.constants import *

from PIL import Image, ImageTk
import project

import keras.backend as K



def process_input(filename_L, filename_R):
    """
    # Function to handle button click event

    :param filename_L: string, path of the left file
    :param filename_R: string, path of the right file
    :return: None
    """
    print(f'paths:\n-{filename_L}\n-{filename_R}')
    global modelcheck
    if not modelcheck:
        print('getting model')
        project.GetModel('./model.h5')
        modelcheck = True

    same, score = project.GetPredict(filename_L, filename_R)
    # same=True

    prediction = 'same author' if same else 'different authors'
    # Display the prediction
    prediction_label.config(text=f"Predicted as {prediction} \nwith confidence score equal to {score} ")
    if not same:
        prediction_label.config(fg="red")
    else:
        prediction_label.config(fg="green")



def select_file(left):
    """
    :param left: boolean
    :return: None
    """
    filename = ""
    filetypes = (
        # ('text files', '*.txt'),
        ('Images Files', ('*.tif', '*.png', '*.jpg', '*.jpeg')),
        ('All files', '*.*')
    )
    if left:
        global filename_L
        filename_L = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes
        )
        filename = filename_L
        image_entry = image_entry_L
        image_label = image_label_L
    else:
        global filename_R
        filename_R = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes
        )
        filename = filename_R
        image_entry = image_entry_R
        image_label = image_label_R

    # Set the selected image path in the entry field
    image_entry.delete(0, tk.END)
    image_entry.insert(0, filename)

    # Display the left image
    image = Image.open(filename)
    image = image.resize((300, 300))  # Adjust the size as desired
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo


# -- MAIN --
modelcheck = False

# Create the Tkinter application
fontstyle = ('Brush Script MT', 12)
# app = tk.Tk()
app = tb.Window(themename='cosmo')
app.configure(bg="white")

app.title("Calliprint - App for Writer Identification")
app.iconbitmap(r".\font\calliprint.ico")


banner_frame = tk.Frame(app, bg="white")
banner_frame.grid(row=0, column=0, columnspan=2, sticky='nswe')


# Load and display the logo in the banner
logo = Image.open(r".\font\calliprint.png").resize((205, 163))

# logo = tk.PhotoImage(file=r".\font\calliprint.png", )
logo = ImageTk.PhotoImage(logo)
# logo_label = tk.Label(banner_frame, image=logo, bg="white")
logo_label = tk.Label(app, image=logo, bg="white")
logo_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)


image_label_L = tk.Label(app)
image_label_R = tk.Label(app)

image_entry_L = tk.Entry(app)
image_entry_R = tk.Entry(app)

open_button_L = tk.Button(app, text='Choose the left handwritten image', command=lambda: select_file(left=True), font=fontstyle, bg='white', fg='dark blue')
open_button_R = tk.Button(app, text='Choose the right handwritten image', command=lambda: select_file(left=False), font=fontstyle)

prediction_label = tk.Label(app, text="Prediction will be displayed here", font=fontstyle)

predict_button = tk.Button(app, text="Predict", command=lambda: process_input(filename_L, filename_R), font=fontstyle)


# defining grid
app.columnconfigure(0, weight=1)
app.columnconfigure(1, weight=1)
app.rowconfigure(0, weight=1)
app.rowconfigure(1, weight=1)
app.rowconfigure(2, weight=1)
app.rowconfigure(3, weight=1)
app.rowconfigure(4, weight=1)
app.rowconfigure(5, weight=1)

# placing grid
# image_label_L.grid(row=1, column=0, sticky=W, pady=2)
image_label_L.grid(row=1, column=0, pady=2)
image_label_R.grid(row=1, column=1, pady=2)
# image_entry_L.grid(row=2, column=0, sticky=W, pady=2)
image_entry_L.grid(row=2, column=0)
image_entry_R.grid(row=2, column=1)
open_button_L.grid(row=3, column=0, pady=2)
open_button_R.grid(row=3, column=1, pady=2)
prediction_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5) #, rowspan = 2
predict_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5) #, rowspan = 2

# Start the Tkinter event loop
app.mainloop()

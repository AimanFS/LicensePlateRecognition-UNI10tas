import tkinter as tk

root = tk.Tk()

canvas1 = tk.Canvas(root, width = 640, height = 480, relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text="Which department is this?")
label1.config(font=('helvetica', 14))
canvas1.create_window(315, 30, window=label1, anchor="center")

label2 = tk.Label(root, text='Type the location:')
label2.config(font=('helvetica', 12))
canvas1.create_window(315, 110, window=label2, anchor = "center")

entry1 = tk.Entry(root)
canvas1.create_window(315,140, window=entry1, anchor = "center")

def gateName():
    x1 = entry1.get()
    
    label1 = tk.Label(root, text = x1)
    canvas1.create_window(315,230,window=label1)

button1 = tk.Button(text="Get the gate open", command=gateName,  bg='brown', fg='white', font=('helvetica', 9, 'bold'))
canvas1.create_window(315, 180, window=button1, anchor = "center")

root.mainloop()
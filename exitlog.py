# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import requests
import mysql.connector
import datetime
from pprint import pprint
import RPi.GPIO as GPIO
import time
from RpiMotorLib import RpiMotorLib
from tkinter import *
import tkinter as tk
import threading
import base64
from urllib.request import urlopen

#Find plate from database
def findplate(plate):
    try:
        sql = "SELECT id,brand,model,color, staff_id FROM vehicles WHERE platenumber = %s AND state=0"
        value = (plate,)
        result = cursor.execute(sql, value)
        record = cursor.fetchone()
        vehicle_id = record[0]
        brand = record[1]
        model = record[2]
        colour = record[3]
        staff_id = record[4]
        graphic.setBrand(brand)
        graphic.setModel(model)
        graphic.setColour(colour)
        if(staff_id != None):
            print("Vehicle found")
            attendancequery(vehicle_id, staff_id)
        else:
            print("Vehicle not registered")
    except Exception as e:
        print("Vehicle not found")
        print(e)


def clockedin(vehicle_id, staff_id):
    date = datetime.datetime.now()
    date_time = date.strftime("%Y-%m-%d %H:%M:%S")
    sql = "SELECT name, username FROM staffs WHERE id = %s"
    value = (staff_id,)
    result = cursor.execute(sql, value)
    record = cursor.fetchone()
    staffname = record[0]
    staffno = record[1]
    graphic.setName(staffname)
    graphic.setID(staffno) 
    try: 
        graphic.setTime(date_time)
        graphic.setLocation(location)
        sql = """INSERT INTO attendances (timein,locationin,created_at,updated_at,vehicle_id, staff_id) VALUES (%s,%s,%s,%s,%s,%s)"""
        val = (date_time, location,date_time,date_time, vehicle_id, staff_id,)
        send = cursor.execute(sql, val)
        mydb.commit()
        print("You've clocked in")
        graphic.setMessage("You've clocked in")
        print(date_time)
        return cursor.lastrowid
    except:
        print("Clock in failed")

#Find date
def attendancequery(vehicle_id, staff_id):
    date = datetime.datetime.now()
    date_time = date.strftime("%Y-%m-%d")
    sql = "SELECT * FROM attendances WHERE DATE(timein) = %s AND vehicle_id = %s"
    value = (date_time, vehicle_id)
    result = cursor.execute(sql, value)
    record = cursor.fetchone()
    if not record:
        print("You haven't logged in")
        print("Creating a log")
        graphic.setMessage("No logs, creating a log")
        attendance_id = clockedin(vehicle_id, staff_id)
        print("Log created")
        clockedout(attendance_id, staff_id)
    else:
        attendance_id = record[0]
        print(attendance_id)
        clockedout(attendance_id, staff_id)
        
#Clocking out attendance
def clockedout(attendance_id, staff_id):
    date = datetime.datetime.now()
    date_time = date.strftime("%Y-%m-%d %H:%M:%S")
    sql1 = "SELECT name, username FROM staffs WHERE id = %s"
    value1 = (staff_id,)
    result1 = cursor.execute(sql1, value1)
    record1 = cursor.fetchone()
    staffname = record1[0]
    staffno = record1[1]
    graphic.setName(staffname)
    graphic.setID(staffno)
    try:
        graphic.setTime(date_time)
        graphic.setLocation(location)
        sql = "UPDATE attendances SET timeout = %s, locationout = %s, updated_at = %s WHERE id = %s"
        val = (date_time, location, date_time, attendance_id, )
        send = cursor.execute(sql, val)
        mydb.commit()
        gatecontrol()
        print(date_time)
        graphic.setMessage("You've clocked out")
        print("You've clocked out")
    except:
        graphics.setMessage("Cannot clock out")


#To control the gates
def gatecontrol():

    # call the function pass the parameters
    graphic.setMessage("Opening the gate")
    mymotortest.motor_run(GpioPins , .001, 128, True, False, "half", .05)
    time.sleep(2)
    graphic.setMessage("Closing the gate")
    mymotortest.motor_run(GpioPins , .001, 128, False, False, "half", .05)
    time.sleep(2)
    graphic.setPlate("")
    graphic.setName("")
    graphic.setID("")
    graphic.setBrand("")
    graphic.setModel("")
    graphic.setColour("")
    graphic.setTime("")
    graphic.setLocation("")
    graphic.setMessage("")

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

class Gui(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
    def callback(self):
        self.root.quit()
    def run(self):
        #initialize the gui window
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)
        self.root.title("UNITEN TAS v3.0")
        #canvas creation
        self.canvas1 = tk.Canvas(self.root, width = 800, height = 600, bg = 'white')
        self.canvas1.pack()
        #System logo
        url="https://i.imgur.com/zdpNaEN.png"
        image_byt = urlopen(url).read()
        image_b64 = base64.encodestring(image_byt)
        photo = tk.PhotoImage(data=image_b64)
        self.canvas1.create_image(400,15,image=photo,anchor='n')
        #connection status label
        self.connectLabel = tk.Label(self.root, text='Connecting', bg='white', fg="green")
        self.connectLabel.config(font=('helvetica', 15))
        self.canvas1.create_window(700, 580, window=self.connectLabel)
        #plate label
        self.plateLabel1 = tk.Label(self.root, text='Plate number:', bg='white')
        self.plateLabel1.config(font=('helvetica', 40))
        self.canvas1.create_window(200, 200, window=self.plateLabel1)
        #plate number label
        self.plateLabel2 = tk.Label(self.root, text='', bg='white')
        self.plateLabel2.config(font=('helvetica', 35))
        self.canvas1.create_window(400, 175, window=self.plateLabel2, anchor='nw')
        #staff label
        self.staffLabel1 = tk.Label(self.root, text='Staff Name:', bg='white')
        self.staffLabel1.config(font=('helvetica', 30))
        self.canvas1.create_window(140, 255, window=self.staffLabel1)
        #staff name label
        self.staffLabel2 = tk.Label(self.root, text='', bg='white')
        self.staffLabel2.config(font=('helvetica', 25))
        self.canvas1.create_window(250, 235, window=self.staffLabel2, anchor='nw')
        #staff id label
        self.idLabel1 = tk.Label(self.root, text='Staff ID No:', bg='white')
        self.idLabel1.config(font=('helvetica', 30))
        self.canvas1.create_window(140, 300, window=self.idLabel1)
        #staff id no label
        self.idLabel2 = tk.Label(self.root, text='', bg='white')
        self.idLabel2.config(font=('helvetica', 30))
        self.canvas1.create_window(250, 277, window=self.idLabel2, anchor='nw')
        #brand label
        self.brandLabel1 = tk.Label(self.root, text='Brand:', bg='white')
        self.brandLabel1.config(font=('helvetica', 30))
        self.canvas1.create_window(95, 350, window=self.brandLabel1)
        #brand name label
        self.brandLabel2 = tk.Label(self.root, text='', bg='white')
        self.brandLabel2.config(font=('helvetica', 30))
        self.canvas1.create_window(175, 325, window=self.brandLabel2, anchor='nw')
        #model label
        self.modelLabel1 = tk.Label(self.root, text='Model:', bg='white')
        self.modelLabel1.config(font=('helvetica', 30))
        self.canvas1.create_window(95, 400, window=self.modelLabel1)
        #model name label
        self.modelLabel2 = tk.Label(self.root, text='', bg='white')
        self.modelLabel2.config(font=('helvetica', 30))
        self.canvas1.create_window(175, 375, window=self.modelLabel2, anchor='nw')
        #colour label
        self.colourLabel1 = tk.Label(self.root, text='Colour:', bg='white')
        self.colourLabel1.config(font=('helvetica', 30))
        self.canvas1.create_window(100, 450, window=self.colourLabel1)
        #colour name label
        self.colourLabel2 = tk.Label(self.root, text='', bg='white')
        self.colourLabel2.config(font=('helvetica', 30))
        self.canvas1.create_window(175, 425, window=self.colourLabel2, anchor='nw')
        #time in label
        self.timeLabel1 = tk.Label(self.root, text='Time out:', bg='white')
        self.timeLabel1.config(font=('helvetica', 30))
        self.canvas1.create_window(115, 500, window=self.timeLabel1)
        #time label
        self.timeLabel2 = tk.Label(self.root, text='', bg='white')
        self.timeLabel2.config(font=('helvetica', 30))
        self.canvas1.create_window(200, 477, window=self.timeLabel2, anchor='nw')
        #location label
        self.locationLabel1 = tk.Label(self.root, text='Location:', bg='white')
        self.locationLabel1.config(font=('helvetica', 30))
        self.canvas1.create_window(120, 550, window=self.locationLabel1)
        #location name label
        self.locationLabel2 = tk.Label(self.root, text='', bg='white')
        self.locationLabel2.config(font=('helvetica', 30))
        self.canvas1.create_window(220, 527, window=self.locationLabel2, anchor='nw')
        #message label
        self.messageLabel1 = tk.Label(self.root, text='Status:', bg='white')
        self.messageLabel1.config(font=('helvetica', 30))
        self.canvas1.create_window(275, 150, window=self.messageLabel1, anchor='center')
        #message type label
        self.messageLabel2 = tk.Label(self.root, text='', bg='white')
        self.messageLabel2.config(font=('helvetica', 30))
        self.canvas1.create_window(525, 150, window=self.messageLabel2, anchor='center')
        self.root.mainloop()
    def update(self):
        self.connectLabel['text'] = "Connected"   
    def setPlate(self, plate):
        self.plateLabel2['text'] = plate
    def setName(self, staffname):
        self.staffLabel2['text'] = staffname
    def setID(self, staffno):
        self.idLabel2['text'] = staffno
    def setBrand(self,brand):
        self.brandLabel2['text'] = brand
    def setModel(self,model):
        self.modelLabel2['text'] = model
    def setColour(self, colour):
        self.colourLabel2['text'] = colour
    def setTime(self, time):
        self.timeLabel2['text'] = time
    def setLocation(self, locationname):
        self.locationLabel2['text'] = locationname
    def setMessage(self, message):
        self.messageLabel2['text'] = message
        
#User input to specify the location it is being scanned
root = tk.Tk()
root.title("Uniten TAS v3.0 Department")

# Gets the requested values of the height and widht.
windowWidth = root.winfo_reqwidth()
windowHeight = root.winfo_reqheight()
 
# Gets both half the screen width/height and window width/height
positionRight = int(root.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(root.winfo_screenheight()/2 - windowHeight/2)
 
# Positions the window in the center of the page.
root.geometry("+{}+{}".format(positionRight, positionDown))

# Add a grid
mainframe = Frame(root)
mainframe.grid(column=0,row=0, sticky=(N,W,E,S) )
mainframe.columnconfigure(0, weight = 1)
mainframe.rowconfigure(0, weight = 1)
mainframe.pack(pady = 100, padx = 100)

# Create a Tkinter variable
tkvar = StringVar(root)

# Dictionary with options
choices = { 'CCI','COE','CES'}
tkvar.set('CCI') # set the default option

popupMenu = OptionMenu(mainframe, tkvar, *choices)
Label(mainframe, text="Select department").grid(row = 1, column = 1)
popupMenu.grid(row = 2, column =1)

# on change dropdown value
def getDepartment(*args):
    global location
    location = tkvar.get()
    root.destroy()

button1 = tk.Button(text='Set department', command=getDepartment)
button1.place(relx=0.5, rely=0.7, anchor=CENTER)
root.mainloop()

MODEL_NAME = "TFlite_model"
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"
min_conf_threshold = float(0.5)
imW, imH = int(640), int(480)

graphic = Gui()
# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])


interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
detectStart = True
#Connect to database
try:
    print("Connecting to database")
    mydb = mysql.connector.connect(
        host="192.168.0.196",
        user="uni10tas",
        passwd="",
        database="uni10tas"
    )
    print("Connected to database")
    cursor = mydb.cursor()
except:
    print("Failed to connect to database")
    detectStart = False

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

GpioPins = [17, 18, 27, 22]

# Declare an named instance of class pass a name and motor type
mymotortest = RpiMotorLib.BYJMotor("MyMotorOne", "28BYJ")

#counter before capturing image
counter = 0
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
if(detectStart == True):
    graphic.update()
    while True:

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame = cv2.flip(frame, -1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                if (object_name == "car"):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    
                    label = '' # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    
                    #capture car data
                    if (object_name == "car"):
                        counter = counter + 1
                        graphic.setMessage("Vehicle detected")
                        print(counter)
                        if (counter > 3):
                            cv2.imwrite('car.jpg',frame)
                            regions = ['my'] # Change to your country
                            with open('car.jpg', 'rb') as fp:
                                response = requests.post(
                                    'https://api.platerecognizer.com/v1/plate-reader/',
                                    data=dict(regions=regions),  # Optional
                                    files=dict(upload=fp),
                                    headers={'Authorization': 'Token fdf4baa4f23f037588abbff0d3a029313ad78765'})
                            result = response.json()
                            if(result != None):
                                try:
                                    plate = result['results'][0]['plate'].upper()
                                    print(plate)
                                    graphic.setMessage("License plate found")
                                    graphic.setPlate(plate)
                                    findplate(plate)
                                except IndexError:
                                    print("Plate not found")
                                    graphic.setMessage("Plate not found")
                            else:
                                print("No plate found")
                                graphic.setMessage("Plate not found")
                            counter = 0
                            
                        
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
mydb.close()
# good practise to cleanup GPIO at some point before exit
GPIO.cleanup()


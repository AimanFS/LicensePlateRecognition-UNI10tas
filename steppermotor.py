import RPi.GPIO as GPIO
import time
# import the library
from RpiMotorLib import RpiMotorLib
    
GpioPins = [17, 18, 27, 22]

# Declare an named instance of class pass a name and motor type
mymotortest = RpiMotorLib.BYJMotor("MyMotorOne", "28BYJ")

# call the function pass the parameters
mymotortest.motor_run(GpioPins , .001, 128, False, False, "half", .05)
time.sleep(2)
mymotortest.motor_run(GpioPins , .001, 128, True, False, "half", .05)

# good practise to cleanup GPIO at some point before exit
GPIO.cleanup()
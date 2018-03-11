import socket
import struct
import cv2
import io
import math
import picamera
from picamera.array import PiRGBArray
import RPi.GPIO as gpio
from keyboard import rc_car
import numpy as np
import time
import pyttsx


#socket client to recv key_vlaues
sock_client=socket.socket()
host='192.168.43.204'
port=5000
sock_client.connect((host,port))
s=1



global engine

#cap=cv2.VideoCapture(1)
print("camera intialised")
'''camera=picamera.PiCamera()
camera.resolution=(320,240)
raw_cap=PiRGBArray(camera)
camera.capture(raw_cap,format="bgr")'''

#gpio.setmode(gpio.Board)
#voice init
engine=pyttsx.init()

#socket to send streme....#
def video_stream(host1,port1):
    global raw_cap
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host1,port1))
    connection = client_socket.makefile('wb')

    try:
        with picamera.PiCamera() as camera:
            camera.resolution = (320, 240)  # pi camera resolution
            camera.framerate = 10  # 10 frames/sec
            time.sleep(2)  # give 2 secs for camera to initilize
            raw_cap = PiRGBArray(camera)
            camera.capture(raw_cap, format="bgr")
            start = time.time()
            stream = io.BytesIO()

            # send jpeg format video stream
            for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                connection.write(struct.pack('<L', stream.tell()))
                connection.flush()
                stream.seek(0)
                connection.write(stream.read())
                if time.time() - start > 600:
                    break
                stream.seek(0)
                stream.truncate()
        connection.write(struct.pack('<L', 0))
    finally:
        connection.close()
        client_socket.close()

def ult_init():
    global ult_trig,ult_echo
    ult_echo=23
    ult_trig=24
    gpio.setmode(gpio.BOARD)
    gpio.setup(ult_trig,gpio.OUT)
    gpio.setup(ult_echo,gpio.IN)
class DistanceToCamera(object):

    def __init__(self):
        # camera params
        self.alpha = 8.0 * math.pi / 180
        self.v0 = 119.865631204
        self.ay = 332.262498472

    def calculate(self,v, h,x_shift,y_shift, image):
        # compute and return the distance from the target point to the camera
        d = h / math.tan(self.alpha + math.atan((v - self.v0) / self.ay))
        if d > 0:
            cv2.putText(image, "%.1fcm" % d,
                ( x_shift,y_shift), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return d
  
        
def ultrasonic():
    ult_init()
    gpio.output(ult_trig,gpio.LOW)
    gpio.output(ult_trig,gpio.HIGH)
    time.sleep(0.1)
    gpio.output(ult_trig,gpio.LOW)

def measure():
    #.....image_processing...#
    # h1: stop sign
    h1 = 15.5 - 10  # cm
    # h2: traffic light
    h2 = 15.5 - 10

    #light_cascade = cv2.CascadeClassifier('xml/traffic_light.xml')
    
    r=0
    d_to_camera = DistanceToCamera()
    # print(d_to_camera)
    while True:
        image=raw_cap.array
        #ret,image=cap.read()
    
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    
         #.......detection of stop sign
        cascade_classifier = cv2.CascadeClassifier("xml/stop_sign.xml")
        
        cascade_obj = cascade_classifier.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
        #            flags=cv2.CV_HAAR_SCALE_IMAGE
                    flags=0
                )
        
                # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (255, 255, 255), 2)
            v = y_pos + height - 5
           
            #print(x_pos+5, y_pos+5, x_pos+width-5, y_pos+height-5, width, height)
    
            # stop sign
            if width/height == 1:
                cv2.putText(image, 'STOP', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                d_to_camera.calculate(v,h1,x_pos+10,y_pos-10,image)
                print("stop")
                rc_car.stop()
                r=1
            else:
                rc_car.forward()
                r=0
    
       #.....detection of traffic light......#
       
       
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        #....Green color....#
        lower_green = np.array([45, 50, 60])
        upper_green = np.array([75, 255, 255])
    
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_green = cv2.erode(mask_green, None, iterations = 6)
        mask_green= cv2.dilate(mask_green, None, iterations = 6)
    
        #....yellow color.....#
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
    
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_yellow = cv2.erode(mask_yellow, None, iterations=4)
        mask_yellow = cv2.dilate(mask_yellow, None, iterations=4)
    
    
        #  For red.....#
        lower_red = np.array([100, 65, 65])
        upper_red = np.array([180, 255, 255])
    
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_red = cv2.erode(mask_red, None, iterations=6)
        mask_red = cv2.dilate(mask_red, None, iterations=6)

        #...contour for green....#
        _,contours,hierarchy = cv2.findContours(mask_green.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            bx,by,bw,bh = cv2.boundingRect(cnt)
            im=cv2.rectangle(image,(bx,by),(bx+bw,by+bh),(0,0,0),3) 
            cv2.putText(im,'green',(bx,by), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
            print("Green light")
        # ..contour for red...#
        if r==0:
            _,contours1,hierarchy1 = cv2.findContours(mask_red.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours1:
                bx,by,bw,bh = cv2.boundingRect(cnt)
                v=by+bh-5
                im=cv2.rectangle(image,(bx,by),(bx+bw,by+bh),(0,0,0),3) 
                cv2.putText(im,'red',(bx,by), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
                d_to_camera.calculate(v,h2,bx+20,by+20,im)
                print("Red light")
                rc_car.stop()
        else:
            r=0
    
    
         #.....contours for yellow...#
        _, contours2, hierarchy2 = cv2.findContours(mask_yellow.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours2:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            im = cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (0, 0, 0), 3) 
            cv2.putText(im, 'Yellow', (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            print("Yellow light")
            time.sleep(5)
            rc_car.forward()
        #....ultrasonic detect and measure
        ultrasonic()
        while (gpio.input(ult_echo) == 0):
            start_pulse = time.time()
        while (gpio.input(ult_echo) == 1):
            end_pulse = time.time()
        duration = end_pulse - start_pulse
        distance = duration * 17160
        distance = round(distance, 2)
        print("distance in centimeter",distance)

        if distance<5:
            print("object is infront off")
            rc_car.stop()
            engine.say("please take of the obstacles")
            engine.runAndWait()
        else:
            rc_car.forward()

        # .....RC car control......#
        while s:
            data = sock_client.recv(1024)
            cmd = data.decode('ascii')

            if cmd == 'left':
                rc_car.left()
                print("moving left")
            if cmd == 'right':
                rc_car.right()
                print("moving right")
            if cmd == 'forward':
                rc_car.forward()
                print("moving forward")
            if cmd == 'backward':
                rc_car.backward()
                print("moving backward")
            if cmd == 'rf':
                rc_car.right_forward()
                print("moving right forward")
            if cmd == 'lf':
                rc_car.left_forward()
                print("moving left forward")
            if cmd == 'rb':
                rc_car.right_backward()
                print("moving right backward")
            if cmd == 'lb':
                rc_car.left_backward()
                print("moving left backward")

        if cv2.waitKey(1) & 0xff==ord('q'):
            break
        
        
measure()
video_stream('192.168.43.204',5000)
cv2.destroyAllWindows()
gpio.cleanup()

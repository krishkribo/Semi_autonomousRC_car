#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 20:35:17 2017

@author: krishna
"""

import numpy as np
import cv2
import socket
import time
import math
import socketserver
import threading
import pygame




class DistanceToCamera(object):

    def __init__(self):
        # camera params
        self.alpha = 8.0 * math.pi / 180
        self.v0 = 119.865631204
        self.ay = 332.262498472

    def calculate(self, v, h,x_shift,y_shift, image):
        # compute and return the distance from the target point to the camera
        d = h*5 / math.tan(self.alpha + math.atan((v - self.v0) / self.ay))
        if d > 0:
            cv2.putText(image, "%.1fcm" % d,
                ( x_shift,y_shift), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return d
    
    
class controller(socketserver.BaseRequestHandler):
   
    pygame.init()

    window = pygame.display.set_mode((800, 600))

    pygame.display.set_caption("Window")


    gameLoop = True

    def handle(self):
        global dstp,rg
        try:
            l = 'left'
            r = 'right'
            f = 'forward'
            b = 'backward'
            wd = 'rf'
            wa = 'lf'
            sd = 'rb'
            sa = 'lb'
            stp = 'stop'
            msg='none'
            while True:
                
                for event in pygame.event.get():
                    
                    if dstp=='dstop':
                        self.request.send(dstp.encode('utf-8'))
                        
                    else:
                        self.request.send(msg.encode('utf-8'))
                        
                    if event.type == pygame.KEYUP:
                        if (event.key == pygame.K_LEFT):
                            self.request.send(l.encode('utf-8'))
                            print('left')


                        if (event.key == pygame.K_RIGHT):
                            self.request.send(r.encode('utf-8'))
                            print("right")


                        if (event.key == pygame.K_UP):
                            self.request.send(f.encode('utf-8'))
                            print("up")


                        if (event.key == pygame.K_DOWN):
                            self.request.send(b.encode('utf-8'))
                            print("down")


                        if event.key == pygame.K_w:
                            self.request.send(wd.encode('utf-8'))
                            print("right_forward")


                        if event.key == pygame.K_a:
                            self.request.send(wa.encode('utf-8'))
                            print("left_forward")


                        if event.key == pygame.K_s:
                            self.request.send(sd.encode('utf-8'))
                            print("right_backward")


                        if event.key == pygame.K_d:
                            self.request.send(sa.encode('utf-8'))
                            print("left_backward")


                        if event.key == pygame.K_x:
                            self.request.send(stp.encode('utf-8'))
                            print("stopped")

                        
        finally:
            print("connection close on thread ")

class CollectTrainingData(socketserver.StreamRequestHandler):
    global h1,h2,d_to_camera
    # collect images for training
    print ('Start collecting images...')
    # h1: stop sign
    h1 = 15.5 - 10  # cm
    # h2: traffic light
    h2 = 15.5 - 10

    # light_cascade = cv2.CascadeClassifier('xml/traffic_light.xml')


    d_to_camera = DistanceToCamera()
    print(d_to_camera)

    def handle(self):
        global h1,h2,d_to_camera,dstp,gr,rg,yd
        r=0
        stream_bytes = ' '
        stream_bytes=stream_bytes.encode('utf-8')
        dstp=0
        rg=0
        print(type(stream_bytes))
        try:

            while True:
                stream_bytes += self.rfile.read(1024)
                #print(stream_bytes)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
        #                    print(jpg)
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),1)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                    # .......detection of stop sign
                    cascade_classifier = cv2.CascadeClassifier("xml/stop_sign.xml")

                    cascade_obj = cascade_classifier.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=0
                    )

                    # draw a rectangle around the objects
                    for (x_pos, y_pos, width, height) in cascade_obj:
                        cv2.rectangle(image, (x_pos + 5, y_pos + 5), (x_pos + width - 5, y_pos + height - 5),
                                      (255, 255, 255), 2)
                        v = y_pos + height - 5

                        # print(x_pos+5, y_pos+5, x_pos+width-5, y_pos+height-5, width, height)

                        # stop sign
                        if width / height == 1:
                            cv2.putText(image, 'STOP', (x_pos, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                                        2)
                            d_to_camera.calculate(v, h1, x_pos + 100, y_pos - 10, image)
                            print("stop")
                            dstp="dstop"
                            #controller.request.send(dstp.encode('utf-8'))
                            r = 1
                        else:
                            dstp=0
                            r = 0

                    # .....detectio of traffic light......#

                    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                    # ....yellow color....#
                    lower_green = np.array([105, 100, 51])
                    upper_green = np.array([125, 255, 255])

                    mask_green = cv2.inRange(hsv, lower_green, upper_green)
                    mask_green = cv2.erode(mask_green, None, iterations=6)
                    mask_green = cv2.dilate(mask_green, None, iterations=6)

                    # ....green color.....#
                    lower_yellow = np.array([54,94,82])
                    upper_yellow = np.array([74, 255, 255])

                    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
                    mask_yellow = cv2.erode(mask_yellow, None, iterations=4)
                    mask_yellow = cv2.dilate(mask_yellow, None, iterations=4)

                    #  For red.....#
                    lower_red = np.array([348,68, 60])
                    upper_red = np.array([368, 255, 255])

                    mask_red = cv2.inRange(hsv, lower_red, upper_red)
                    mask_red = cv2.erode(mask_red, None, iterations=6)
                    mask_red = cv2.dilate(mask_red, None, iterations=6)

                    '''for (x_pos, y_pos, width, height) in :
                        cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (255, 255, 255), 2)
                        v = y_pos + height - 5'''

                    # ...contour for green....#
                    _, contours, hierarchy = cv2.findContours(mask_green.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        im = cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (0, 0, 0), 3)
                        cv2.putText(im, 'yellow', (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        print("yellow light")
                        gr="green detected"
                        #self.request.send(gr.encode('utf-8'))
                    # ..contour for red...#
                    if r == 0:
                        _, contours1, hierarchy1 = cv2.findContours(mask_red.copy(), cv2.RETR_TREE,
                                                                    cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours1:
                            bx, by, bw, bh = cv2.boundingRect(cnt)
                            v = by + bh - 5
                            im = cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (0, 0, 0), 3)
                            cv2.putText(im, 'red', (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                        cv2.LINE_AA)
                            d_to_camera.calculate(v, h2, bx + 20, by + 20, im)
                            print("Red light")
                            rg="red"
                            #self.request.send(rg.encode('utf-8'))
                    else:
                        rg=0
                        r = 0
                        # .....contours for yellow...#
                    _, contours2, hierarchy2 = cv2.findContours(mask_yellow.copy(), cv2.RETR_TREE,
                                                                cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours2:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        im = cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (0, 0, 0), 3)
                        cv2.putText(im, 'Green', (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                    cv2.LINE_AA)
                        print("Green light")
                        yd="yellow"
                        #self.request.send(yd.encode('utf-8'))

                    cv2.imshow('image', image)

                    if cv2.waitKey(1) & 0xff==ord('q'):
                        break
            cv2.destroyAllWindows()
        finally:
            print("connection close")


class ThreadServer(object):

    def server_thread(host, port):
        server = socketserver.TCPServer((host, port), CollectTrainingData)
        print("server connection established on port 1")
        server.serve_forever()

    def server_thread2(host, port):
        server = socketserver.TCPServer((host, port),controller)
        print("server connection established on port 2")
        server.serve_forever()

    video_thread = threading.Thread(target=server_thread, args=('192.168.43.204', 5000))
    video_thread.start()
    controller_thread = threading.Thread(target=server_thread2, args=('192.168.43.204',4000))
    controller_thread.start()


if __name__ == '__main__':
    ThreadServer()

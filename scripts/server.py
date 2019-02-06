#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from threading import Thread
import http.server
import socketserver
import os
import sys 

global PORT
global DIR

def servidor():    
    global PORT
    global DIR
        
    web_dir = DIR
    os.chdir(web_dir)
    
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", PORT), Handler)
    print("serving at port", PORT, DIR)
    httpd.serve_forever()

PORT = int(sys.argv[1])
DIR = sys.argv[2]

threaded = Thread(target=servidor, args=()).start()

for i in range(10):
    print("passou", i)
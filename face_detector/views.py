# -*- coding: utf-8 -*-
"""
Created on May 15, 2017

@author: Hua Zheng
"""
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib2
import json
import math
import cv2
import os
import sys
import dlib
from face_detector.models import Shopper
from keras.engine import Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
import pylibmc
from memcached_stats import MemcachedStats
import redis

r = redis.StrictRedis(host="web-app-staging.wfh6et.0001.use1.cache.amazonaws.com", port=6379)
PUSHER_CHANNEL = "PI-Pusher-In"
CHANNEL = "pi-go-experiment"
PRODUCT_ON = "product-on-shelf"
PRODUCT_OFF = "product-off-shelf"
PRODUCT_WELCOME = "new-shopper"
PUSHER_MESSAGE = "{\"event\":\"%s\", \"channel\":\"%s\", \"message\": \"%s\"}"

#import mxnet as mx
#from face_detector.MTCNNDetector.mtcnn_detector import MtcnnDetector

#detector = dlib.get_frontal_face_detector()
# define the path to the face detector
#FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
#    base_path=os.path.abspath(os.path.dirname(__file__)))
#FACE_DETECTOR_PATH = '/Users/hua.zheng/Documents/Project/VisualRecognition/go/haarcascade_frontalface_default.xml'
vgg_model = VGGFace()  # pooling: None, avg or max
out = vgg_model.get_layer('fc7').output
vgg_model_fc7 = Model(vgg_model.input, out)
detector = dlib.get_frontal_face_detector()
mc = pylibmc.Client(["127.0.0.1"], binary=True,behaviors={"tcp_nodelay": True,"ketama": True})
mem = MemcachedStats()

@csrf_exempt
def detect(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}
    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)
            uuid = request.POST.get("uuid", None)
            if uuid is None:
                uuid = "a"

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)
            #image = image.transpose((1, 0, 2))
            max_right = image.shape[1]
            max_bottom = image.shape[0]

        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        #rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
        #                                  minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        # MTCNN face detector
        cropped = False
        dets = detector(image, 1)
        if len(dets) == 0:
            data["error"] = "No face detected."
            return JsonResponse(data)
        print("Number of faces detected: {}".format(len(dets)))
        '''
        if len(dets) == 0:
            new_top = image.shape[0]*0.15
            new_bottom = image.shape[0]*0.85
            new_left = image.shape[1]*0.15
            new_right = image.shape[1]*0.85
            image = image[new_top:new_bottom, new_left:new_right]
            dets = detector(image, 1)
            cropped = True
            if len(dets) == 0:
                data["error"] = "No face detected."
                return JsonResponse(data)
        '''
        for i, d in enumerate(dets):
            if i == 0:
                top = max(d.top(), 0)
                bottom = min(d.bottom(), max_bottom)
                left = max(d.left(), 0)
                right = min(d.right(), max_right)
                roi_color = image[top:bottom, left:right]
                x = cv2.resize(roi_color, (224, 224))
                x = np.expand_dims(x, axis=0).astype(np.float32)
                ftr = vgg_model_fc7.predict(x)
                if uuid in mc:
                    pre_img_feature = mc.get(uuid)
                    pre_img_feature.append(ftr)
                    mc[uuid] = pre_img_feature
                    print uuid,'has',len(pre_img_feature), 'images stored'
                else:
                    mc[uuid] = [ftr]
                    print uuid,'new customer'
        '''
        detector = MtcnnDetector(model_folder='/Users/hua.zheng/cv_api/face_detector/MTCNNDetector/model',
                                 ctx=mx.cpu(0),num_worker=4, accurate_landmark=False)
        results = detector.detect_face(image)

        if results is not None:
            print '----------'
            print 'find chip!'
            total_boxes = results[0]
            points = results[1]
            if len(total_boxes) == 0:
                data["error"] = "No face detected."
                return JsonResponse(data)
            # extract aligned face chips
            chips = detector.extract_image_chips(image, points, 224, 0.37)

            for i, chip in enumerate(chips):
                if i == 0:
                    shopperList[uuid].append(chip)
                    print '----------'
                    print 'find chip!'
                cv2.imwrite('/Users/hua.zheng/cv_api/chip_' + str(i) + '.jpg', chip)
        '''
        # construct a list of bounding boxes from the detection
        rects = [(int(d.left()), int(d.top()), int(d.right()), int(d.bottom())) for d in dets]
        # update the data dictionary with the faces detected
        r.publish(PUSHER_CHANNEL, PUSHER_MESSAGE
                  % (PRODUCT_WELCOME, CHANNEL, uuid+";"+url))
        data.update({"uuid": uuid,"total_num_faces": len(dets),"faces": rects, "success": True})
        #del vgg_model_fc7, vgg_model, out
        # return a JSON response
    return JsonResponse(data)

@csrf_exempt
def getId(request):
    # initialize the data dictionary to be returned by the request
    data = {'uuid': 'None'}
    # if rect passed, then detect the img.
    StartDetector = False
    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)
            left = request.POST.get("left", None)
            right = request.POST.get("right", None)
            top = request.POST.get("top", None)
            bottom = request.POST.get("bottom", None)
            action = request.POST.get("action", None)
            # if the URL is None, then return an error
            if left is None:
                StartDetector = True
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)
            max_right = image.shape[1]
            max_bottom = image.shape[0]
            print max_right,max_bottom,StartDetector,top,left,bottom,right
        ### START WRAPPING OF COMPUTER VISION APP
        if StartDetector:
            dets = detector(image, 1)
            print 'detected'
            if len(dets) == 0:
                data["error"] = "No face detected."
                print("Number of faces detected: {}".format(len(dets)))
                return JsonResponse(data)
            for i, d in enumerate(dets):
                if i == 0:
                    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                        i, d.left(), d.top(), d.right(), d.bottom()))
                    top = max(d.top(), 0)
                    bottom = min(d.bottom(), max_bottom)
                    left = max(d.left(), 0)
                    right = min(d.right(),max_right)

        else:
            top = max(int(top),0)
            print top
            bottom = min(int(bottom),max_bottom)
            left = max(int(left),0)
            right = min(int(right),max_right)

        roi_color = image[top:bottom, left:right]
        x = cv2.resize(roi_color, (224, 224))
        x = np.expand_dims(x, axis=0).astype(np.float32)
        try:
            ftr = vgg_model_fc7.predict(x)
        except Exception as err:
            print err

        # compare the input img with cached imgs
        try:
            shopperKeys = mem.keys()
            minDiff = 10000000.0
            best_match = ""
            diffDic = {}
        except Exception as err:
            print err

        for key in shopperKeys:
            cachedFeatures = mc.get(key)
            diff = 0.0
            for cachedFtr in cachedFeatures:
                # cosine similarity or norm2 similarity
                #diff = dcos(cachedFeature.astype(np.float32),ftr.astype(np.float32))
                diff += math.sqrt(sum([(cachedFtr[0][i]-ftr[0][i])**2 for i in range(len(ftr[0]))]))
            avg_msdiff = diff / len(cachedFeatures)
            diffDic[key] = avg_msdiff
            if minDiff > avg_msdiff:
                best_match = key
                minDiff = avg_msdiff
        confidence = 1.0 - minDiff/sum(diffDic.values())

        ### END WRAPPING OF COMPUTER VISION APP

        # update the data dictionary
        #data.update({"uuid": best_match, "match_difference": minDiff,"confidence": confidence,"success": True})
        data.update({"uuid": best_match, "match_difference": minDiff,"confidence": confidence,"difference2Keys": str(diffDic) ,"success": True})
    # return a JSON response
    r.publish(PUSHER_CHANNEL, PUSHER_MESSAGE
              % (action, CHANNEL, best_match))
    return JsonResponse(data)

def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)

    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib2.urlopen(url)
            data = resp.read()

        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()

        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        print image, len(image)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        print image.shape
        resp.close()
    # return the image
    return image

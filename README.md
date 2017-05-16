# pi-go
This is a partial project from PI-GO which is an experimental and amazon-go-like project started by some software engineers and data scientists in Pointinside, Derek, Shi, Hua Zheng, John Hansler, Tom Walsh, Wenwen Song. In PI-Go, We try to combine the sensor and computer vision to create a "walk-in-walk-out" store setup.

The whole project is still under development but the first demo has been demoed on Monday, May 15, 2017. 

"face-detect-compare" project aims to create a facial detection and matching server to (1) cache customers' uuid and facial features; (2) return the uuid for any image of customers' faces taken when they are taking the product from the shelf. This process will be trigered by the sensor.

Functionality
=======
There are two main functionality for the project.

1. get the uuid and image url from mobile scan app, detect the face and cache the face with the uuid. 

2. get the image url and detect and compare the image with cached images and send back its uuid. 

How to use
=======
```
 //For example: 
 // suppose deploy the server on ec2-54-165-64-229.compute-1.amazonaws.com:8080
 // to check in the new customers with uuid and image url, POST:
 $ curl -X POST 'http://ec2-54-165-64-229.compute-1.amazonaws.com:8080/face_detection/detect/' -d 'url=https://s3.amazonaws.com/pointinside-public/hackathon-pi-go/captures/08190e7c-a313-4351-b285-8a695d91177a.jpg' -d 'uuid=derek';
 $ {"total_num_faces": 1, "uuid": "derek", "success": true, "faces": [[538, 365, 761, 588]]}
 // to get the uuid of a customer when he/she took or return an item, POST: 
 $ curl -X POST 'http://ec2-54-165-64-229.compute-1.amazonaws.com:8080/face_detection/getId/' -d 'url=https://s3.amazonaws.com/pointinside-public/hackathon-pi-go/captures/08190e7c-a313-4351-b285-8a695d91177a.jpg';
 $ {"uuid": derek, "match_difference": 503.23,"confidence": 0.88,"difference2Keys": str(diffDic) ,"success": True}
```

How to deploy the server
=======
before starting the server, you might need to install bunch of packages. You can find them in DeployServer.sh.
```
// start the memcached server which is by default ruuning on 127.0.0.1:11211 Port.
sudo service memcached start
// start the server
python manage.py runserver 0.0.0.0:8080
```
Real-time face detection and feature extractors
=======
1. For face detector we apply dlib '_face_detector' but a mxnet face detector MTCNN is also provided. We might add more reliable and accurate face detector later on.
 
	* dlib: http://dlib.net/
	
	* mtcnn: https://github.com/pangyupo/mxnet_mtcnn_face_detection

2. For face decriptor, we use
 
	* vgg face: refer to "http://www.robots.ox.ac.uk/~vgg/software/vgg_face/"
	
	* vgg keras implementation: https://github.com/rcmalli/keras-vggface

3. For the feature comparison, we do not do too much embedding except extracting the second last fc layer before the output layer and compare iamge by Root-Mean-Square distance. We are also planning to reudce the dimension by t-SNE later on to improve the accuracy.

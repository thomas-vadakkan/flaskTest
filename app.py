from sys import stdout
from makeup_artist import Makeup_artist
import logging
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from camera import Camera
from utils import base64_to_pil_image, pil_image_to_base64
import cv2
import numpy as np


app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app)
camera = Camera(Makeup_artist())

thres = 0.5
nms_threshold = 0.2

className = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    className = f.read().rsplit('\n')
    # print(className)


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'


# settings
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# @socketio.on('input image', namespace='/test')
# def test_message(input):
#     input = input.split(",")[1]
#     camera.enqueue_input(input)
#     image_data = input # Do your magical Image processing here!!
#     #image_data = image_data.decode("utf-8")
#     image_data = "data:image/jpeg;base64," + image_data
#     print("OUTPUT " + image_data)
#     emit('out-image-event', {'image_data': image_data}, namespace='/test')
    #camera.enqueue_input(base64_to_pil_image(input))


@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""

    app.logger.info("starting to generate frames!")
    while True:
        success, img  = camera.get_frame() #pil_image_to_base64(camera.get_frame())
                    classIDs, confs, bbox= net.detect(img, confThreshold=thres)
            # print(classIDs,bbox)

            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1, -1)[0])
            confs = list(map(float, confs))
            indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)



            if len(classIDs) !=0:
                count = [0] * len(className)
                for i in indices:
                    i = i[0]
                    box = bbox[i]
                    x,y,w,h = box[0],box[1],box[2],box[3]

                    cv2.rectangle(img,(x,y),(x+w,h+y),color=(0,255,0),thickness=2)

                    cv2.putText(img,className[classIDs[i][0]-1].upper(),(box[0]+10,box[1]+30),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confs[i]*100,2))+"%",(box[0]+200,box[1]+30),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                    for m in classIDs[i]:
                        count[m] = count[m] + 1


                cv2.putText(img,className[classIDs[i][0]-1]+": "+str(count[m]),(50,50+m),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                field_ref.set({
                    className[classIDs[i][0]-1]: str(count[m])
                }, merge=True)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app)

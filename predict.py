import os
import cv2
import sys
import time
import math
import getopt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import *
from glob import glob
from parser import parser
from TrackNet import ResNet_Track
from focal_loss import BinaryFocalLoss
from collections import deque
from tensorflow import keras

args = parser.parse_args()
tol = args.tol
mag = args.mag
sigma = args.sigma
HEIGHT = args.HEIGHT
WIDTH = args.WIDTH
BATCH_SIZE = 1
FRAME_STACK = args.frame_stack
load_weights = args.load_weights
video_path = args.video_path
csv_path = args.csv_path
preds_path = args.preds_path
#preds_name = args.preds_name
results_vid_path = args.results_vid_path

opt = keras.optimizers.legacy.Adadelta(learning_rate=1.0)
model = ResNet_Track(input_shape=(3, HEIGHT, WIDTH))
model.compile(loss=BinaryFocalLoss(gamma=2), optimizer=opt, metrics=[keras.metrics.BinaryAccuracy()])
try:
    model.load_weights(load_weights)
    print("Load weights successfully")
except:
    print("Fail to load weights, please modify path in parser.py --load_weights")

if not os.path.isfile(video_path) or not video_path.endswith('.mp4'):
    print("Not a valid video path! Please modify path in parser.py --video_path")
    sys.exit(1)
else:
    # acquire video info
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.split(video_path)[-1][:-4]

# if not os.path.isfile(csv_path) and not csv_path.endswith('.csv'):
#     compute = False
info = {
    idx: {
        'Frame': idx,
        'Ball': 0,
        'x': -1,
        'y': -1
    } for idx in range(n_frames)
}
    # print("Predict only, will not calculate accurracy")

print('Beginning predicting......')

# img_input initialization
gray_imgs = deque()
success, image = cap.read()
ratio = image.shape[0] / HEIGHT

size = (int(WIDTH * ratio), int(HEIGHT * ratio))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

if not os.path.exists(results_vid_path):
    os.mkdir(results_vid_path)

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = np.expand_dims(img, axis=2)
gray_imgs.append(img)
for _ in range(FRAME_STACK - 1):
    success, image = cap.read()

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=2)
    gray_imgs.append(img)

frame_no = FRAME_STACK - 1
time_list = []
x_predictions = []
y_predictions = []
while success:
    if frame_no == n_frames:
        break
    img_input = np.concatenate(gray_imgs, axis=2)
    img_input = cv2.resize(img_input, (WIDTH, HEIGHT))
    img_input = np.moveaxis(img_input, -1, 0)
    img_input = np.expand_dims(img_input, axis=0)
    img_input = img_input.astype('float') / 255.

    start = time.time()
    y_pred = model.predict(img_input, batch_size=BATCH_SIZE)
    end = time.time()
    time_list.append(end - start)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype('float32')
    y_true = []
    if info[frame_no]['Ball'] == 0:
        y_true.append(genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag))
    else:
        y_true.append(
            genHeatMap(WIDTH, HEIGHT, int(info[frame_no]['x'] / ratio), int(info[frame_no]['y'] / ratio), sigma, mag))

    h_pred = y_pred[0] * 255
    h_pred = h_pred.astype('uint8')
    if np.amax(h_pred) <= 0:
        x_predictions.append(None)
        y_predictions.append(None)
    else:
        cnts, _ = cv2.findContours(h_pred[0].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]
        max_area_idx = 0
        max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
        for i in range(1, len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        target = rects[max_area_idx]
        print('FRRRRAAAME')
        (cx_pred, cy_pred) = (int(ratio * (target[0] + target[2] / 2)), int(ratio * (target[1] + target[3] / 2)))
        x_predictions.append(cx_pred)
        y_predictions.append(cy_pred)

    success, image = cap.read()
    if success:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
        gray_imgs.append(img)
        gray_imgs.popleft()
        frame_no += 1

xy_predictions = list(zip(x_predictions, y_predictions))
df = pd.DataFrame(xy_predictions, columns=['X-coordinate', 'Y-coordinate'])

if not os.path.exists(preds_path):
    os.mkdir(preds_path)

df.to_csv(str(preds_path + video_name + '.csv'))

# out.release()
total_time = sum(time_list)



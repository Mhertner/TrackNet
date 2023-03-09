import cv2
import pandas as pd

from parser import parser
# deque is a list with fast appends and pops
from collections import deque
import numpy as np

class CreateVideo:

    def __init__(self):
        # self.args = parser.parse_args()
        # self.HEIGHT = self.args.HEIGHT
        # self.WIDTH = self.args.WIDTH
        # self.args = parser.parse_args()
        self.HEIGHT = 288
        self.WIDTH = 512

    def run(self, video_path, csv_path, output_path, video_name):

        """
        Takes an input csv, containing the coordination of the ball,
        and puts them on the input video. Output is a saved video in a specified folder
        """

        cap = cv2.VideoCapture(video_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))

        data = pd.read_csv(csv_path)

        gray_imgs = deque()
        success, image = cap.read()
        ratio = image.shape[0] / self.HEIGHT
        #
        size = (int(self.WIDTH * ratio), int(self.HEIGHT * ratio))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #
        out = cv2.VideoWriter(output_path + video_name + '_predict.mp4', fourcc, fps, size)
        #
        out.write(image)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
        gray_imgs.append(img)

        # for _ in range(FRAME_STACK - 1):
        #     success, image = cap.read()
        #     out.write(image)
        #
        #     img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #     img = np.expand_dims(img, axis=2)
        #     gray_imgs.append(img)

        for i, row in data.iterrows():
            success, image = cap.read()
            # print(i, row)
            if (row[1], row[2]) == (0,0):
                image_cp = np.copy(image)
                out.write(image_cp)
                continue
            print(i, row[1], row[2])
            # img_input = np.concatenate(gray_imgs, axis=2)
            # img_input = cv2.resize(img_input, (self.WIDTH, self.HEIGHT))
            # img_input = np.moveaxis(img_input, -1, 0)
            # img_input = np.expand_dims(img_input, axis=0)
            # img_input = img_input.astype('float') / 255.

            image_cp = np.copy(image)
            cv2.circle(image_cp, (int(row[1]), int(row[2])), 5, (0, 0, 255), -1)
            out.write(image_cp)
        out.release

        return None

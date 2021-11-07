import cv2
import sys
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import dlib
import numpy as np
import math
import time
import pywinauto

SHAPE_DETECTOR = "shape_predictor_68_face_landmarks.dat"
PREVIEW_MOUTH_COLOUR = (0, 255, 0)
MAR_THRESHOLD = 1.5
(MOUTH_LM_INDEX_START, MOUTH_LM_INDEX_END) = (48, 68)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_DETECTOR)
MOUTH_OPEN_MESSAGE = 'open'
MOUTH_CLOSE_MESSAGE = 'close'
PREV_STATE = MOUTH_CLOSE_MESSAGE
KEY_TO_PRESS = '{F14}'


def distance(point1, point2):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return math.sqrt(dx * dx + dy * dy)


def landmark_shape_to_np(lm_shape):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((lm_shape.num_parts, 2), dtype="int")

    # convert all facial landmarks to tuples of (x, y)-coordinates
    for i in range(0, lm_shape.num_parts):
        coordinates[i] = (lm_shape.part(i).x, lm_shape.part(i).y)

    return coordinates


def mouth_aspect_ratio(mouth_landmarks):
    # calculate the vertical distances of 2 vertical mouth landmarks
    vertical1_d = distance(
        mouth_landmarks[2], mouth_landmarks[10])  # indexes 51, 59
    vertical2_d = distance(
        mouth_landmarks[4], mouth_landmarks[8])  # indexes 53, 57

    # calculate the horizontal distance of horizontal mouth landmarks
    horizontal_d = distance(
        mouth_landmarks[0], mouth_landmarks[6])  # indexes 49, 55

    # calculate the mouth aspect ratio
    mar = (vertical1_d + vertical2_d) / horizontal_d
    return mar


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    changeLabel = pyqtSignal(str)

    def run(self):
        global PREV_STATE
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)
                if(len(rects) == 1):
                    rect = rects[0]  # add trycatch here
                    shape = predictor(gray, rect)
                    shape = landmark_shape_to_np(shape)
                    mouth = shape[MOUTH_LM_INDEX_START:MOUTH_LM_INDEX_END]
                    hull = cv2.convexHull(mouth)
                    mar = mouth_aspect_ratio(mouth)
                    if mar > MAR_THRESHOLD:
                        # mouth is open
                        if PREV_STATE == MOUTH_CLOSE_MESSAGE:
                            self.changeLabel.emit(MOUTH_OPEN_MESSAGE)
                        PREV_STATE = MOUTH_OPEN_MESSAGE
                        # time.sleep(3)
                    else:
                        # mouth is closed

                        if PREV_STATE == MOUTH_OPEN_MESSAGE:
                            self.changeLabel.emit(MOUTH_CLOSE_MESSAGE)
                        PREV_STATE = MOUTH_CLOSE_MESSAGE
                        # self.changeLabel.emit(MOUTH_CLOSE_MESSAGE)

                else:
                    print(f'error with face detect, len(rects) {len(rects)}')
                    time.sleep(1)
                    # self.changeLabel.emit(f'Error: Number of faces found: {len(rects)}')
                self.changePixmap.emit(p)


class App(QWidget):
    def __init__(self):
        # TODO: Trigger only after closing +open cycle is complete
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 600
        self.initUI()
        self.communicator_app = None

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def findCommunicatorApp(self):
        app_list = pywinauto.findwindows.find_elements()
        for app_ in app_list:
            if 'Communicator' in app_.rich_text:
                app = pywinauto.application.Application()
                app.connect(handle=app_.handle)
                app.top_window().set_focus()
                return app

    def updateMouthOpenLabel(self, text):
        print(f'Received {text}')
        if(text == MOUTH_OPEN_MESSAGE):
            if(self.communicator_app == None):
                self.communicator_app = self.findCommunicatorApp()
                self.communicator_app.top_window().type_keys(KEY_TO_PRESS)
                print("Sent F5 press")
            else:
                print(f'app {self.communicator_app} found')
                self.communicator_app.top_window().type_keys(KEY_TO_PRESS)
                print("Sent F5 press")

            # pyautogui.press('space')
            self.mouthOpenLabel.setText(text + "clicked space")
        else:
            pass

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(700, 700)
        # create a label
        self.label = QLabel(self)
        self.label.move(10, 10)
        self.label.resize(640, 480)
        self.mouthOpenLabel = QLabel(self)
        self.mouthOpenLabel.move(10, 500)
        self.mouthOpenLabel.resize(400, 200)
        self.mouthOpenLabel.setAlignment(Qt.AlignCenter)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.changeLabel.connect(self.updateMouthOpenLabel)
        self.communicator_app = self.findCommunicatorApp()
        th.start()
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

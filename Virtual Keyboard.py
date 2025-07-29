import cv2 as cv
import cv2_ext
import HandTrackingModule as htm
import numpy as np
import time
from pynput.keyboard import Controller


keyboard = cv.imread('keyboard.png', cv.IMREAD_UNCHANGED)

# Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
KEYBOARD_HEIGHT, KEYBOARD_WIDTH = keyboard.shape[:2]
KEY_SIZE = 75
PADDING_X = PADDING_Y = 10


cap = cv.VideoCapture(0)
cap.set(3, WINDOW_WIDTH)
cap.set(4, WINDOW_HEIGHT)
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (WINDOW_WIDTH, WINDOW_HEIGHT))
detector = htm.HandDetector(max_num_hands=1, min_tracking_confidence=0.7)

previous_time = 0
frame_counter = 0

text = ""

controller = Controller()

keys = [["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

last_pressed_key = None
last_press_time = 0
key_cooldown = 0.4

if not cap.isOpened():
    print("Error: Could not open the camera.")
else:
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv.flip(frame, 1)

        if frame_counter % 2 == 0:
            handInfo, frame = detector.findHands(frame, draw=False)

        keyboard_pos = ((WINDOW_WIDTH - KEYBOARD_WIDTH) // 2, (WINDOW_HEIGHT - KEYBOARD_HEIGHT - 30))

        cv.putImage(frame, keyboard, keyboard_pos)

        if handInfo:
            handLMS = handInfo[0]['lmList']
            thumb_tip, index_tip = handLMS[4][:2], handLMS[8][:2]
            cv.circle(frame, (thumb_tip[0], thumb_tip[1]), 8, (255, 0, 255), -1)
            cv.circle(frame, (index_tip[0], index_tip[1]), 8, (255, 0, 255), -1)


            if (keyboard_pos[0] < thumb_tip[0] < keyboard_pos[0] + KEYBOARD_WIDTH and
                keyboard_pos[1] < thumb_tip[1] < keyboard_pos[1] + KEYBOARD_HEIGHT
            ):
                ref_dist, _ = detector.getPixelDistance(handLMS[5][:2], handLMS[17][:2])
                pinch = detector.getActualDistance(thumb_tip, index_tip, ref_dist, 7) < 1.8

                if pinch:
                    col = (thumb_tip[0] - keyboard_pos[0]) // KEY_SIZE
                    row = (thumb_tip[1] - keyboard_pos[1]) // KEY_SIZE

                    if 0 <= row < len(keys) and 0 <= col < len(keys[row]):
                        key = keys[row][col]
                        current_time = time.time()

                        if key != last_pressed_key or (current_time - last_press_time) > key_cooldown:
                            last_pressed_key = key
                            last_press_time = current_time

                            text += key
                            controller.press(key)

        (TEXT_WIDTH, TEXT_HEIGHT), BASELINE = cv.getTextSize(text, cv.FONT_HERSHEY_PLAIN, 5, 3)

        center_y = 200

        top_left = (
            max((WINDOW_WIDTH - TEXT_WIDTH) // 2 - PADDING_X, 10),
            center_y - TEXT_HEIGHT - PADDING_Y // 2
        )
        bottom_right = (
            min((WINDOW_WIDTH + TEXT_WIDTH) // 2 + PADDING_X, WINDOW_WIDTH - 10),
            center_y + PADDING_Y + BASELINE
        )

        if text != "":
            (TEXT_WIDTH, TEXT_HEIGHT), BASELINE = cv.getTextSize(text, cv.FONT_HERSHEY_PLAIN, 5, 3)
            center_x = WINDOW_WIDTH // 2

            cv.roundRect(frame, top_left, bottom_right, 20, (0, 0, 0), -1)

            text_pos = (
                center_x - TEXT_WIDTH // 2,
                center_y + TEXT_HEIGHT // 2
            )
            cv.putText(frame, text, text_pos, cv.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)


        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv.putText(frame, f'FPS: {int(fps)}', (10, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        
        out.write(frame)

        cv.imshow('Live Cam', frame)

        frame_counter += 1

        if cv.waitKey(1) == 27:
            break

cap.release()
out.release()
cv.destroyAllWindows()
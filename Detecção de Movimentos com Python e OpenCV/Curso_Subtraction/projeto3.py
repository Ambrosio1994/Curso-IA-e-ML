import numpy as np
import cv2
import sys
import time
import validator
from random import randint

LINE_IN_COLOR = (64, 255, 0)
LINE_OUT_COLOR = (0, 0, 255)
BOUNDING_BOX_COLOR = (255, 128, 0)
TRACKER_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
CENTROID_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
TEXT_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
TEXT_POSITION_BGS = (10, 50)
TEXT_POSITION_COUNT_CARS = (10, 100)
TEXT_POSITION_COUNT_TRUCKS = (10, 150)
TEXT_SIZE = 1.2
FONT = cv2.FONT_HERSHEY_SIMPLEX
SAVE_IMAGE = True
IMAGE_DIR = "./veiculos"
VIDEO_SOURCE = "videos/Traffic_3.mp4"
VIDEO_OUT = "videos/results/result_traffic.avi"

BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
BGS_TYPE = BGS_TYPES[2]

def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if KERNEL_TYPE == "opening":
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == "closing":
        kernel = np.ones((11, 11), np.uint8)

    return kernel

def getFilter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)

    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)

    if filter == 'dilation':
        return cv2.dilate(img, getKernel("dilation"), iterations=2)

    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)
        dilation = cv2.dilate(opening, getKernel("dilation"), iterations=2)

        return dilation

def getBGSubtractor(BGS_TYPE):
    if BGS_TYPE == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=.8)
    if BGS_TYPE == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=.7, noiseSigma=0)
    if BGS_TYPE == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=False, varThreshold=200)
    if BGS_TYPE == "KNN":
        return cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=400, detectShadows=True)
    if BGS_TYPE == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True,
                                                        maxPixelStability=15 * 60, isParallel=True)
    print("Detector inválido")
    sys.exit(1)

def getCentroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return (cx, cy)

#print(getCentroid(50, 100, 100, 100))

def save_frame(frame, file_name, flip=True):
    if flip: # BGR -> RGB
        cv2.imwrite(file_name, np.flip(frame, 2))
    else:
        cv2.imwrite(file_name, frame)

cap = cv2.VideoCapture(VIDEO_SOURCE)
hasFrame, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer_video = cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

# ROI
bbox = cv2.selectROI(frame, False)
#print(bbox)
(w1, h1, w2, h2) = bbox

frameArea = h2 * w2
#print(frameArea)
minArea = int(frameArea / 250)
maxArea = 15000
#print(minArea)

line_IN = int(h1)
line_OUT = int(h2 - 20)
#print(line_IN, line_OUT)

DOWN_limit = int(h1 / 4)
print('Down IN limit Y', str(DOWN_limit))
print('Down OUT limit Y', str(line_OUT))

bg_subtractor = getBGSubtractor(BGS_TYPE)

def main():
    frame_number = -1
    cnt_cars, cnt_trucks = 0, 0
    objects = []
    max_p_age = 2
    pid = 1

    while (cap.isOpened()):

        ok, frame = cap.read()
        if not ok:
            print("Erro")
            break

        roi = frame[h1:h1 + h2, w1:w1 + w2]

        for i in objects:
            #print('teste')
            i.age_one()

        frame_number += 1
        bg_mask = bg_subtractor.apply(roi)
        bg_mask = getFilter(bg_mask, 'combine')
        (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Contagem dos carros
            if area > minArea and area <= maxArea:
                x,y,w,h = cv2.boundingRect(cnt)
                centroid = getCentroid(x, y, w, h)
                cx = centroid[0]
                cy = centroid[1]
                new = True
                cv2.rectangle(roi, (x, y), (x + 50, y - 13), TRACKER_COLOR, -1)
                cv2.putText(roi, 'CAR', (x, y - 2), FONT, 0.5, (255,255,255), 1, cv2.LINE_AA)
                for i in objects:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        if i.going_DOWN(DOWN_limit) == True:
                            cnt_cars += 1
                            if SAVE_IMAGE:
                                save_frame(roi, IMAGE_DIR + '/car_DOWN_%04d.png' % frame_number)
                                print('ID:', i.getId(), ' passou pela estrada em', time.strftime('%c'))
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > line_OUT:
                            i.setDone()
                    if i.timedOut():
                        index = objects.index(i)
                        objects.pop(index)
                        del i
                if new == True:
                    p = validator.MyValidator(pid, cx, cy, max_p_age)
                    objects.append(p)
                    pid += 1
                cv2.circle(roi, (cx, cy), 5, CENTROID_COLOR, -1)

            # Contando caminhões
            elif area >= maxArea:
                x, y, w, h = cv2.boundingRect(cnt)
                centroid = getCentroid(x, y, w, h)
                cx = centroid[0]
                cy = centroid[1]

                new = True

                cv2.rectangle(roi, (x, y), (x + 50, y - 13), TRACKER_COLOR, -1)
                cv2.putText(roi, 'TRUCK', (x, y - 2), FONT, .5, (255, 255, 255), 1, cv2.LINE_AA)

                for i in objects:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        if i.going_DOWN(DOWN_limit) == True:
                            cnt_trucks += 1
                            if SAVE_IMAGE:
                                save_frame(roi, IMAGE_DIR + "/truck_DOWN_%04d.png" % frame_number, flip=False)
                                print("ID:", i.getId(), 'passou pela estrada', time.strftime("%c"))
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > line_OUT:
                            i.setDone()
                    if i.timedOut():
                        index = objects.index(i)
                        objects.pop(index)
                        del i
                if new == True:
                    p = validator.MyValidator(pid, cx, cy, max_p_age)
                    objects.append(p)
                    pid += 1
                cv2.circle(roi, (cx, cy), 5, CENTROID_COLOR, -1)

        for i in objects:
            cv2.putText(roi, str(i.getId()), (i.getX(), i.getY()), FONT, 0.3, TEXT_COLOR, 1, cv2.LINE_AA)

        str_cars = 'Cars: ' + str(cnt_cars)
        str_trucks = 'Trucks: ' + str(cnt_trucks)

        frame = cv2.line(frame, (w1, line_IN), (w1 + w2, line_IN), LINE_IN_COLOR, 2)
        frame = cv2.line(frame, (w1, h1 + line_OUT), (w1 + w2, h1 + line_OUT), LINE_OUT_COLOR, 2)

        cv2.putText(frame, str_cars, TEXT_POSITION_COUNT_CARS, FONT, 1, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(frame, str_cars, TEXT_POSITION_COUNT_CARS, FONT, 1, (232, 162, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, str_trucks, TEXT_POSITION_COUNT_TRUCKS, FONT, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, str_trucks, TEXT_POSITION_COUNT_TRUCKS, FONT, 1, (232, 162, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, 'Background Subtractor: ' + BGS_TYPE, TEXT_POSITION_BGS, FONT, TEXT_SIZE, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Background Subtractor: ' + BGS_TYPE, TEXT_POSITION_BGS, FONT, TEXT_SIZE, (128, 0, 255), 2, cv2.LINE_AA)

        # Sobreposições transparentes
        # https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
        for alpha in np.arange(0.3, 1.1, 0.9)[::-1]:
            overlay = frame.copy()
            output = frame.copy()
            cv2.rectangle(overlay, (w1, h1), (w1 + w2, h1 + h2), BOUNDING_BOX_COLOR, -1)
            frame = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', bg_mask)

        writer_video.write(frame)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()


main()













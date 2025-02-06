import numpy as np
import cv2
import sys

TEXT_COLOR = (0, 255, 0)
TRACKER_COLOR = (255, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = "videos/Animal_1.mp4"

BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
BGS_TYPE = BGS_TYPES[2]

def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if KERNEL_TYPE == "opening":
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == "closing":
        kernel = np.ones((3, 3), np.uint8)

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
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if BGS_TYPE == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    if BGS_TYPE == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print("Detector inválido")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_SOURCE)
minArea = 250
bg_subtractor = getBGSubtractor(BGS_TYPE)

def main():
    while (cap.isOpened):

        ok, frame = cap.read()
        if not ok:
            print("Erro")
            break

        frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

        bg_mask = bg_subtractor.apply(frame)
        bg_mask = getFilter(bg_mask, 'combine')
        bg_mask = cv2.medianBlur(bg_mask, 5)

        (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(contours)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= minArea:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (10,30), (250,55), (255,0,0), -1)
                cv2.putText(frame, 'Movimento detectado!', (10,50), FONT, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

                cv2.drawContours(frame, cnt, -1, TRACKER_COLOR, 3)
                cv2.drawContours(frame, cnt, -1, (255,255,255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), TRACKER_COLOR, 3)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 1)

                # Sobreposições transparentes
                # https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
                #for alpha in np.arange(0.8, 1.1, 0.9)[::-1]:
                #    frame_copy = frame.copy()
                #    output = frame.copy()
                #    cv2.drawContours(frame_copy, [cnt], -1, TRACKER_COLOR, -1)
                #    frame = cv2.addWeighted(frame_copy, alpha, output, 1 - alpha, 0, output)

        result = cv2.bitwise_and(frame, frame, mask=bg_mask)
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

main()

















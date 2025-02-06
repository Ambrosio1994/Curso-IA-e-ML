import numpy as np
import cv2

VIDEO_SOURCE = 'videos/Cars.mp4'
VIDEO_OUT = 'videos/results/filtragem_mediana_temporal.avi'

cap = cv2.VideoCapture(VIDEO_SOURCE)
hasFrame, frame = cap.read()
#print(hasFrame, frame.shape)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (frame.shape[1], frame.shape[0]), False)

#print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#print(np.random.uniform(size=25))
framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
#print(framesIds)

# 108, 2000
#cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)
#hasFrame, frame = cap.read()
#cv2.imshow('Teste', frame)
#cv2.waitKey(0)

frames = []
for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame = cap.read()
    frames.append(frame)

#print(np.asarray(frames).shape)
#print(frames[0])
#print(frames[1])

#for frame in frames:
#    cv2.imshow('Frame', frame)
#    cv2.waitKey(0)

#print(np.mean([1, 3, 5, 6, 8, 9]))
#print((1 + 3 + 5 + 6 + 8 + 9) / 6)
#print(np.median([1, 3, 5, 6, 8, 9]))
#print((5 + 6) / 2)
#print(np.median([1, 3, 4, 5, 6, 8, 9]))

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
#print(frame[0])
#print(medianFrame)
#cv2.imshow('Median frame', medianFrame)
#cv2.waitKey(0)

cv2.imwrite('model_median_frame.jpg', medianFrame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray', grayMedianFrame)
#cv2.waitKey(0)

while (True):
    hasFrame, frame = cap.read()

    if not hasFrame:
        print('Error')
        break

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dframe = cv2.absdiff(frameGray, grayMedianFrame)

    #th, dframe = cv2.threshold(dframe, 70, 255, cv2.THRESH_BINARY)
    th, dframe = cv2.threshold(dframe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(th)

    cv2.imshow('frame', dframe)
    writer.write(dframe)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
cap.release()



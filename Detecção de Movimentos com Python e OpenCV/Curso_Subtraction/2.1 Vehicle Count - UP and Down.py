import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import time

from modules import validator
from random import randint

# Verificar a versão do opencv
major = cv2.__version__.split('.')[0]

# Função abaixo é opcional
# Salvar os registros de cada detecção em arquivo CSV para posterior análise
# Por exemplo, carro 3 foi detectado as 00:00 em 1 de abril de 2021
# https://realpython.com/python-csv/#writing-csv-files-with-csv
# https://docs.python.org/pt-br/3/library/csv.html
import csv
fp = open('result/report.csv', mode='w')        
writer_CSV = csv.DictWriter(fp, fieldnames=['time', 'vehicle'])
writer_CSV.writeheader()

# Variáveis de controle e padronização
# ============================================================================
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
SAVE_IMAGE = False
IMAGE_DIR = "./result/out"
VIDEO_SOURCE = "videos/Traffic_3.mp4"
VIDEO_OUT = "result/result.mp4"

# vetor de background subtractors
BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]

# definir qual BGS usar, inserindo o número que corresponde ao BGS selecionado no vetor de background subtractors
# 0 = GMG, 1 = MOG, 2 = MOG2, 3 = KNN,  4 = CNT
BGS_TYPE = BGS_TYPES[2]

# Kernel: Elemento estruturante
def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if KERNEL_TYPE == "opening":
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == "closing":
        kernel = np.ones((11, 11), np.uint8)

    return kernel

# Filtros morfológicos para redução do ruído
def getFilter(img, filter):
    '''            
        Esses filtros são escolhidos a dedo, apenas com base em testes visuais
    '''
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
        return cv2.createBackgroundSubtractorMOG2(history = 50, detectShadows=False, varThreshold = 200)
    if BGS_TYPE == "KNN":
        return cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=400, detectShadows=True)
    if BGS_TYPE == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True, maxPixelStability=15*60, isParallel=True)
    print("Unknown createBackgroundSubtractor type")
    sys.exit(1)

# Calcular o centro de cada objeto detectado
def getCentroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)

def save_frame(frame, file_name, flip=True):
    # mudar de RGB para BGR
    if flip:
        cv2.imwrite(file_name, np.flip(frame, 2))
    else:
        cv2.imwrite(file_name, frame)

# Carregar o video
cap = cv2.VideoCapture(VIDEO_SOURCE)
hasFrame, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*"MP4V")
writer_VIDEO = cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

# Limites da ROI (Region of Interest)
# Definimos uma região de interesse para diminuir a interferência de outros objetos na imagem.
# Por exemplo, se estiver contando veículos e uma pessoa passar pela lateral da pista, o algoritmo irá detectar o movimento. Agora se focar somente no movimento da pista, onde os veículos passam e "ignorar" o que está fora da ROI não teremos este problema
# h: altura, w: largura
# h1, h2 = 100, 300
# w1, w2 = 350, 900
bbox = cv2.selectROI(frame, False)
(w1, h1, w2, h2) = bbox
# largura, altura, largura, altura
print(bbox)

# Limiar das dimensões do objeto na ROI (-->> valor empírico)
frameArea = h2*w2
minArea = int(frameArea/250)
maxArea = 8000 
print('Area Threshold', minArea)

# Linhas limites de entrada/saída (line_DOWN, line_UP)
# Para que um objeto seja contado, ele terá que entrar pela área de entrada, cruzando a linha de entrada(line_IN) e sair pela área de saída, cruzando a linha de saída (line_UP)
line_DOWN = int(h1+20)

# Linha de saída está posiciona a 20 pixels a menos na vertical, ou seja, ela é mais alta que o limite da ROI
# h2 = 300, então 300-20 = 280 na vertical
line_UP = int(h2-20)

# Limiar de entrada do objeto na ROI na coordenada Y
# O objeto precisa estar dentro deste limiar (-->> valor empírico)
UP_limit = int(h1/5)
DOWN_limit = int(h1/4)

print("UP limit y:", str(UP_limit))
print("DOWN limit y:", str(DOWN_limit))

bg_subtractor = getBGSubtractor(BGS_TYPE)

def main():
    frame_number = -1

    # Para armazenar a contagem de objetos que cruzaram a linha
    cnt_cars, cnt_trucks = 0, 0    

    # Vetor para armazenar os objetos detectados
    objects = []

    # Tempo limite de vida/existência do objeto na classe
    max_p_age = 2

    # Inicialização do ID dos objetos detectados
    pid = 1

    while (cap.isOpened()):

        ok, frame = cap.read()
        if not ok:
          print("Frame capture failed, stopping...")
          break
        
        # Limitar o processamento das imagens à somente a área de interesse (ROI)
        roi = frame[h1:h1 + h2, w1:w1 + w2]
        
        # Percorrer o vetor de objetos
        for i in objects:
            i.age_one()  # marcar cada detecção como um objeto

        # Usamos o número do frame para servir com índice ao salvar os resultados
        frame_number += 1

        # Aplicar o BGS na ROI
        bg_mask = bg_subtractor.apply(roi)

        # Tenta aplicar os filtros morfológicos, caso ocorra alguma erro, retorna 0 (zero) para a contagem de carros e caminhões
        try:
            bg_mask = getFilter(bg_mask, 'combine')            
        except:
            print('CARS:', cnt_cars)
            print('TRUCKS:', cnt_trucks)
            break
        
        # Se a versão do OpenCV for 3, então usa 3 parâmetros
        if major == '3':
            (img, contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else: # Nas versões mais recentes do OpenCV, cv2.findContours() foi alterada para retornar apenas os contornos e a hierarquia
            (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Percorrer os contornos detectados
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Contando carros
            if area > minArea and area <= maxArea:
                x, y, w, h = cv2.boundingRect(cnt)

                # Calcular o centroide dos objetos detectados
                centroid = getCentroid(x, y, w, h)
                cx = centroid[0]
                cy = centroid[1]

                new = True   

                # Desenhar um pequeno retângulo preenchido acima do objeto detectado e escrever qual foi o objeto detectado
                cv2.rectangle(roi, (x, y), (x+50, y-13), TRACKER_COLOR, -1)
                cv2.putText(roi, 'CAR', (x, y-2), FONT, .5, (255,255,255), 1, cv2.LINE_AA)            

                # Percorrer o vetor de objetos
                for i in objects:
                    # O método abs() retorna o valor absoluto do número
                    # Exemplo: 
                    # floating = -30.33
                    # print('O valor absoluto de -30,33 é:', abs(floating))
                    # O valor absoluto de -30,33 é: 30,33
                    
                    # Se o valor absoluto do novo objeto na horizontal (x) menos o valor na horizontal (getX) do objeto na lista de objetos for menor ou igual ao valor na horizontal do objeto detectado 
                    # E Se o valor absoluto do novo objeto na vertical (y) menos o valor na vertical (getY) do objeto na lista de objetos for menor ou igual ao valor na vertical do objeto detectado
                    # Então não é um novo objeto, somente atualizamos as coordenadas
                    if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)
                        # Verificar se o objeto está descendo, ou seja, indo de cima para baixo
                        if i.going_DOWN(DOWN_limit) == True:
                            cnt_cars += 1
                            if SAVE_IMAGE:
                                save_frame(roi, IMAGE_DIR + "/car_DOWN_%04d.png" % frame_number, flip=False) # Salvar a imagem de cada veículo detectado
                                writer_CSV.writerow({'time': time.strftime("%c"), 'vehicle': 'Car DOWN ' + str(cnt_cars)}) # Gravar o resultado em  CSV
                            print("ID:", i.getId(),'crossed going down at', time.strftime("%c"))
                        
                        # Verificar se o objeto está sudingo, ou seja, indo de baixo para cima
                        elif i.going_UP(UP_limit) == True:
                            cnt_cars +=1
                            if SAVE_IMAGE:
                                save_frame(roi, IMAGE_DIR + "/car_UP_%04d.png" % frame_number, flip=False) # Salvar a imagem de cada veículo detectado
                                writer_CSV.writerow({'time': time.strftime("%c"), 'vehicle': 'Car UP ' + str(cnt_cars)}) # Gravar o resultado em  CSV
                            print("ID:", i.getId(),'crossed going up at', time.strftime("%c"))
                        break
                    if i.getState() == '1':
                        # Verificar se o objeto cruzou a linha de saída, vindo de cima para baixo
                        if i.getDir() == 'down' and i.getY() > line_UP:
                            i.setDone()
                        # Verificar se o objeto cruzou a linha de saída, vindo de baixo para cima
                        if i.getDir() == 'up' and i.getY() < line_DOWN:
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
                img = cv2.rectangle(roi, (x, y), (x+w, y+h), TRACKER_COLOR, 2)
            
            # Contando caminhões
            elif area >= maxArea:
                x, y, w, h = cv2.boundingRect(cnt)
                centroid = getCentroid(x, y, w, h)
                cx = centroid[0]
                cy = centroid[1]

                new = True

                cv2.rectangle(roi, (x, y), (x+50, y-13), TRACKER_COLOR, -1)
                cv2.putText(roi, 'TRUCK', (x, y-2), FONT, .5, (255,255,255), 1, cv2.LINE_AA)

                # Percorrer o vetor de objetos
                for i in objects:
                    if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        # Verificar se o objeto está descendo, ou seja, indo de cima para baixo
                        if i.going_DOWN(DOWN_limit) == True:
                            cnt_trucks += 1
                            if SAVE_IMAGE:
                                save_frame(roi, IMAGE_DIR + "/truck_DOWN_%04d.png" % frame_number, flip=False)
                                writer_CSV.writerow({'time': time.strftime("%c"), 'vehicle': 'Truck DOWN ' + str(cnt_trucks)})
                            print("ID:", i.getId(),'crossed going down at', time.strftime("%c"))

                        # Verificar se o objeto está sudingo, ou seja, indo de baixo para cima
                        elif i.going_UP(UP_limit) == True:
                            cnt_trucks +=1
                            if SAVE_IMAGE:
                                save_frame(roi, IMAGE_DIR + "/truck_UP_%04d.png" % frame_number, flip=False) # Salvar a imagem de cada veículo detectado
                                writer_CSV.writerow({'time': time.strftime("%c"), 'vehicle': 'Truck UP ' + str(cnt_trucks)}) # Gravar o resultado em  CSV
                            print("ID:", i.getId(),'crossed going up at', time.strftime("%c"))
                        break
                    if i.getState() == '1':
                        # Verificar se o objeto cruzou a linha de saída, vindo de cima para baixo
                        if i.getDir() == 'down' and i.getY() > line_UP:
                            i.setDone()
                         # Verificar se o objeto cruzou a linha de saída, vindo de baixo para cima
                        if i.getDir() == 'up' and i.getY() < line_DOWN:
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
                img = cv2.rectangle(roi, (x, y), (x+w, y+h), TRACKER_COLOR, 2)

        for i in objects:
            cv2.putText(roi, str(i.getId()), (i.getX(), i.getY()),
                       FONT, 0.3, TEXT_COLOR, 1, cv2.LINE_AA)

        str_cars = 'Cars: ' + str(cnt_cars)
        str_trucks = 'Trucks: ' + str(cnt_trucks)
        
        frame = cv2.line(frame,(w1, line_DOWN),(w1 + w2, line_DOWN),LINE_IN_COLOR,2)
        frame = cv2.line(frame,(w1, h1 + line_UP),(w1 + w2, h1 + line_UP),LINE_OUT_COLOR,2)

        cv2.putText(frame, 'Background Subtractor: ' + BGS_TYPE, TEXT_POSITION_BGS, FONT, TEXT_SIZE, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, 'Background Subtractor: ' + BGS_TYPE, TEXT_POSITION_BGS, FONT, TEXT_SIZE, (128, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, str_cars, TEXT_POSITION_COUNT_CARS, FONT, TEXT_SIZE, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, str_cars, TEXT_POSITION_COUNT_CARS, FONT, TEXT_SIZE, (255, 128, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, str_trucks, TEXT_POSITION_COUNT_TRUCKS, FONT, TEXT_SIZE, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, str_trucks, TEXT_POSITION_COUNT_TRUCKS, FONT, TEXT_SIZE, (255, 128, 0), 2, cv2.LINE_AA)              

        # Sobreposições transparentes
        # https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
        for alpha in np.arange(0.3, 1.1, 0.9)[::-1]:
            overlay = frame.copy()
            output = frame.copy()
            cv2.rectangle(overlay, (w1, h1), (w1 + w2, h1 + h2), BOUNDING_BOX_COLOR, -1)
            frame = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
        # Função para visualizar o modelo fundo
        # Disponível somente para MOG2, KNN, CNT
        if BGS_TYPE != 'MOG' and BGS_TYPE != 'GMG':
            bg = bg_subtractor.getBackgroundImage()
            # cv2.imshow('Model', bg)
        cv2.imshow('Frame', frame)
        cv2.imshow('ROI_Mask', bg_mask)

        if SAVE_IMAGE:
            writer_VIDEO.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    writer_VIDEO.release()
    cap.release()
    cv2.destroyAllWindows()

main()
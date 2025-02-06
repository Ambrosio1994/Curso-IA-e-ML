# Baseado em: https://github.com/sarful/People-counter-opencv-python3/blob/master/Person.py

import math

class MyValidator:
    tracks = []

    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None

    def getTracks(self):
        return self.tracks

    def getId(self):
        return self.i

    def getState(self):
        return self.state

    def getDir(self):
        return self.dir

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def setDone(self):
        self.done = True

    def timedOut(self):
        return self.done

    def going_DOWN(self, mid_start):

        # Verificar se tem pelo menos 2 coordenadas de objetos armazenadas
        if len(self.tracks) >= 2:

            # Verificar se o estado do objeto é zero
            # O estado do objeto só mudará quando cruzar o limiar de entrada
            if self.state == '0':

                # Cáculo da distância euclidiana
                distance = math.sqrt(float((self.tracks[-1][1] - self.tracks[-2][1])**2) + float(
                    (self.tracks[-1][1] - self.tracks[-2][1])**2))
                if distance < 10:
                    # [-2] são duas posições anteriores do registro no vetor e [1] é a coluna contendo os
                    # valores na vertical (y) de cada objeto
                    # Se a posição vertical anterior do objeto for maior que o limiar de entrada e se em
                    # duas posições verticais anteriores o   valor for menor ou igual ao limiar de entrada
                    # Então atualizamos o estado do objeto para 1 e indicamos que o mesmo se moveu na
                    # direção para baixo (down)
                    # Fazemos isso para ter certeza de que o objeto cruzou a linha de entrada, movendo-se
                    # de cima para baixo
                    if self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start:
                        state = '1'
                        self.dir = 'down'
                        return True
            else:
                return False
        else:
            return False

    def going_UP(self, mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                distance = math.sqrt(float((self.tracks[-1][1] - self.tracks[-2][1])**2) + float(
                    (self.tracks[-1][1] - self.tracks[-2][1])**2))
                if distance < 10:
                    if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end:
                        state = '1'
                        self.dir = 'up'
                        return True
            else:
                return False
        else:
            return False

    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True
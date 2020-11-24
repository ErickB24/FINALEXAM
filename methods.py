from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import collections

class final :

    def color(self,image):
        image = np.array(image, dtype=np.float64) / 255  # Normalizar imagen
        rows, cols, ch = image.shape  # Tamaño de la imagen en filas, columnas y capas
        image_array = np.reshape(image, (
        rows * cols, ch))  # Cambiar el tamaño de la imagen con respecto a las filas,columnas y en cual capa
        image_array_sample = shuffle(image_array, random_state=0)[:10000] # Tomar 10k muestras de la matriz de la imagen

        kmeans = KMeans(n_clusters=4, random_state=0).fit(
        image_array_sample)                 # Extraer el modelo kmean con pocas muestras
        self.labels = kmeans.predict(image_array)  # Etiquetas para cada uno de los pixeles
        centros = kmeans.cluster_centers_       # Centros de los clousters
        numero_colors = np.amax(self.labels)+1  # numero maximo de labels

        return numero_colors

    def porcentaje(self,image):
        # cantidad de veces que hay alguna de las etiquetas
        color_1 = collections.Counter(self.labels)[0]
        color_2 = collections.Counter(self.labels)[1]
        color_3 = collections.Counter(self.labels)[2]
        color_4 = collections.Counter(self.labels)[3]

        # Porcentaje de la cantidad de veces
        porcentaje1 = (color_1 / len(self.labels)) * 100
        porcentaje2 = (color_2 / len(self.labels)) * 100
        porcentaje3 = (color_3 / len(self.labels)) * 100
        porcentaje4 = (color_4 / len(self.labels)) * 100

        porcentaje = [porcentaje1, porcentaje2, porcentaje3, porcentaje4]

        #image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #hist_hsv = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])

        return porcentaje

    def direct_HT(self, theta_data):

        rmax = int(round(0.5 * np.sqrt(self.rows ** 2 + self.cols ** 2)))
        # y , x = np.where(M >= 0.1)
        y, x = np.where(self.bw_edges >= 1)

        x_ = x - self.center_x
        y_ = y - self.center_y

        th = theta_data[y, x] + np.pi / 2

        hist_val, bin_edges = np.histogram(th, bins=32)
        print('Histogram', hist_val)

        print(np.amin(th), np.amax(th))
        th[y_ < 0] = th[y_ < 0] + np.pi
        print(np.amin(th), np.amax(th))
        accumulator = np.zeros((rmax, len(self.theta)))

        r = np.around(x_ * np.cos(th) + y_ * np.sin(th))
        r = r.astype(int)
        th = np.around(360 * th / np.pi)
        th = th.astype(int)
        th[th == 720] = 0
        print(np.amin(th), np.amax(th))
        r_idx = np.where(np.logical_and(r >= 0, r < rmax))
        np.add.at(accumulator, (r[r_idx[0]], th[r_idx[0]]), 1)
        return accumulator

    def find_peaks(self, accumulator, nhood, accumulator_threshold, N_peaks):
        done = False
        acc_copy = accumulator
        nhood_center = [(nhood[0] - 1) / 2, (nhood[1] - 1) / 2]
        peaks = []
        while not done:
            [p, q] = np.unravel_index(acc_copy.argmax(), acc_copy.shape)
            if acc_copy[p, q] >= accumulator_threshold:
                peaks.append([p, q])
                p1 = p - nhood_center[0]
                p2 = p + nhood_center[0]
                q1 = q - nhood_center[1]
                q2 = q + nhood_center[1]

                [qq, pp] = np.meshgrid(np.arange(np.max([q1, 0]), np.min([q2, acc_copy.shape[1] - 1]) + 1, 1), \
                                       np.arange(np.max([p1, 0]), np.min([p2, acc_copy.shape[0] - 1]) + 1, 1))
                pp = np.array(pp.flatten(), dtype=np.intp)
                qq = np.array(qq.flatten(), dtype=np.intp)

                acc_copy[pp, qq] = 0
                done = np.array(peaks).shape[0] == N_peaks
            else:
                done = True

        return peaks
    def Haugh(self, bw_edges):
        [self.rows, self.cols] = bw_edges.shape[:2]
        self.center_x = self.cols // 2
        self.center_y = self.rows // 2
        self.theta = np.arange(0, 360, 0.5)
        self.bw_edges = bw_edges
    def orientacion (self,bw_bordes):

        # TRANSFORMADA DE HAUGH
        acc_thresh = 50
        N_peaks = 10
        nhood = [25, 9]

        [self.rows, self.cols] = bw_bordes.shape[:2]
        self.center_x = self.cols // 2
        self.center_y = self.rows // 2
        self.theta = np.arange(0, 360, 0.5)
        self.bw_edges = bw_bordes
        rmax = int(round(0.5 * np.sqrt(self.rows ** 2 + self.cols ** 2)))
        y, x = np.where(self.bw_edges >= 1)

        accumulator = np.zeros((rmax, len(self.theta)))

        for idx, th in enumerate(self.theta):
            r = np.around(
                (x - self.center_x) * np.cos((th * np.pi) / 180) + (y - self.center_y) * np.sin((th * np.pi) / 180))
            r = r.astype(int)
            r_idx = np.where(np.logical_and(r >= 0, r < rmax))
            np.add.at(accumulator[:, idx], r[r_idx[0]], 1)
        done = False

        #PEAKS
        acc_copy = accumulator
        nhood_center = [(nhood[0] - 1) / 2, (nhood[1] - 1) / 2]
        peaks = []
        while not done:
            [p, q] = np.unravel_index(acc_copy.argmax(), acc_copy.shape)
            if acc_copy[p, q] >= acc_thresh:
                peaks.append([p, q])
                p1 = p - nhood_center[0]
                p2 = p + nhood_center[0]
                q1 = q - nhood_center[1]
                q2 = q + nhood_center[1]

                [qq, pp] = np.meshgrid(np.arange(np.max([q1, 0]), np.min([q2, acc_copy.shape[1] - 1]) + 1, 1), \
                                       np.arange(np.max([p1, 0]), np.min([p2, acc_copy.shape[0] - 1]) + 1, 1))
                pp = np.array(pp.flatten(), dtype=np.intp)
                qq = np.array(qq.flatten(), dtype=np.intp)

                acc_copy[pp, qq] = 0
                done = np.array(peaks).shape[0] == N_peaks
            else:
                done = True

            # PARA CADA PICO

            for i in range(len(peaks)):
                rho = peaks[i][0]
                theta_ = self.theta[peaks[i][1]]

                theta_pi = np.pi * theta_ / 180
                theta_ = theta_ - 180

        if (179<=theta_<=181):
            print("vertical")
            angulo = "vertical"
        if (-179>=theta_>=-181):
            print("vertical")
            angulo = "vertical"
        if (89<=theta_<=91):
            print("horizontal")
            angulo = "horizontal"
        if (-89 >= theta_ >= -91):
            print("horizontal")
            angulo="horizontal"

        return angulo
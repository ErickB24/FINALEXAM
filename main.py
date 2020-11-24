from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from methods import final
import os

print('inserte el directorio de la imagen:')    # Mensaje en consola de insertar directorio de la imagen
path = input()                                  # Directorio ingresado por el usuario
path_file = os.path.join(path)                  # Lectura del directorio de la imagen
image = cv2.imread(path_file)                   # Lectura de la imagen
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Re-ordenar para matplot

fn = final()

#NUMERO DE COLORES
n_color = fn.color(image)
print("numero de colores de la bandera: ")
print (n_color)

#PORCENTAJE DE COLOR
p_1, p_2, p_3, p_4 = fn.porcentaje(image)
print("porcentaje color 1: ")
print(p_1)
print("porcentaje color 2: ")
print(p_2)
print("porcentaje color 3: ")
print(p_3)
print("porcentaje color 4: ")
print(p_4)

#ANGULO de Orientacion
high_thresh = 300  # Threshold
bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh, L2gradient=True)
angulo = fn.orientacion(bw_edges)
print(angulo)


# C:\Users\Erick\Desktop\PARCIAL FINAL\flag1.png


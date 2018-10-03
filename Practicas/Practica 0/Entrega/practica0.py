import cv2
from PIL import Image                 # funciones para cargar y manipular imágenes
import numpy as np                # funciones numéricas (arrays, matrices, etc.)
from matplotlib import pyplot as plt
import random

# importamos el modulo pyplot, y lo llamamos plt
import matplotlib.pyplot as plt

#configuracion necesaria de pyplot para ver las imagenes en escala de grises
plt.rcParams['image.cmap'] = 'gray'

#Ej1:
#Escribir una función que lea el fichero de una imagen y la
#muestre tanto en grises como en color ( im=leeimagen(filename,flagColor))

def leeimagen(filename,flagColor):
    i=cv2.imread(filename,flagColor)

    return i

def pintaI(i):
    cv2.imshow('image',i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def pintaMI(vim):
    for n in vim:
        pintaI(n)


def modcolor(im):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            color=im[i,j]
            r=color[0]+20
            g=color[1]-20
            b=color[2]-20

            if(r>255): r=255
            if(g<0): g=0
            if(b<0): b=0

            im[i,j]=[r,g,b]

    cv2.imwrite("img/ej4.jpg",im)
    pintaI(im)


def pintarContatenado(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grey_3 = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    concatenate = np.concatenate((img, grey_3), axis=1)

    cv2.imwrite("img/ej5.jpg",concatenate)
    pintaI(concatenate)


def main():

    #ej1
    im=leeimagen("img/rosalia.jpg",1)

    #ej2
    pintaI(im)



    #ej3
    img1=leeimagen("img/rosalia.jpg",0)
    img2=leeimagen("img/all.jpg",1)
    listaImagenes=[img1,img2]
    pintaMI(listaImagenes)

    #ej4

    print("Intro para comenzar el cambio de colores.")
    input()
    modcolor(im)



    #ej5
    pintarContatenado(im)



main()

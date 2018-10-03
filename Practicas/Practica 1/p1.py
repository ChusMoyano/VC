import cv2
import numpy as np

MAX_KERNEL_LENGTH=11



def concatenate(imagenes):
    conc=np.concatenate(imagenes,axis=1)
    pintaI(conc)

def concatenateCuadrado(imagenes):
    conc=np.concatenate((imagenes[0],imagenes[1]),axis=1)
    conc1=np.concatenate((imagenes[2],imagenes[3]),axis=1)

    concFin=np.concatenate((conc,conc1),axis=0)
    pintaI(concFin)



def leeimagen(filename,flagColor):
    i=cv2.imread(filename,flagColor)

    return i

def pintaI(i):
    cv2.imshow('image',i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Gato
img=leeimagen("cat.bmp",1)
#Rosalia
#img=leeimagen("img/rosalia.jpg",1)
#perro
#img=leeimagen("dog.bmp",1)

#Ejercicio 1
#apartado a

def convolucion(im,k,sig):
    iC=cv2.GaussianBlur(im, (k , k), sig)
    pintaI(iC)
    return iC

def ej1A():
    img0=convolucion(img,5,0)
    img1=convolucion(img,5,1)
    img2=convolucion(img,5,2)
    img3=convolucion(img,5,3)

    concatenate((img0,img1,img2,img3))


def ej1B():
    blur=cv2.blur(img,(5,5),1)

    kx,ky=cv2.getDerivKernels(1,1,3,normalize=False)

    imgSalida=cv2.sepFilter2D(blur,-1,kx,ky,delta=15)
    concatenate((img,imgSalida))



def ej1C():
    deltaParam=50

    img1=cv2.copyMakeBorder(src=img,top=10,bottom=10,left=10,right=10,borderType=1)
    img2=cv2.copyMakeBorder(src=img,top=10,bottom=10,left=10,right=10,borderType=4)

    blur11=cv2.blur(img1,(3,3),1)
    blur12=cv2.blur(img2,(3,3),1)

    blur31=cv2.blur(img1,(3,3),3)
    blur32=cv2.blur(img2,(3,3),3)

    imgOut1=cv2.Laplacian(blur11,ksize=3,ddepth=0,delta=deltaParam)
    imgOut2=cv2.Laplacian(blur12,ksize=3,ddepth=0,delta=deltaParam)
    imgOut3=cv2.Laplacian(blur31,ksize=3,ddepth=0,delta=deltaParam)
    imgOut4=cv2.Laplacian(blur32,ksize=3,ddepth=0,delta=deltaParam)

    concatenate((imgOut1,imgOut2))
    concatenate((imgOut3,imgOut4))




#ej2
imgAux=np.copy(img)

while(True):
    pintaI(imgAux)
    cv2.pyrUp(imgAux,imgAux,dstsize=[1.5,1.5])
    pintaI(imgAux)

    #cv2.pyrDown(imgAux,imgAux,dstsize=0.5)



#

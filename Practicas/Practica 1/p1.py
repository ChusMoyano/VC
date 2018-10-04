import cv2
import numpy as np

MAX_KERNEL_LENGTH=11


def concatenateDifSizes(images):
        height = sum(image.shape[0] for image in images)
        width = max(image.shape[1] for image in images)
        output = np.zeros((height,width,3))

        y = 0
        for image in images:
            h,w,d = image.shape
            output[y:y+h,0:w] = image
            y += h

        #cv2.imwrite("test.jpg", output)
        return output


def pintaMI(vim):
    for n in vim:
        pintaI(n)

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
img=leeimagen("data/cat.bmp",1)

#Ejercicio 1
#apartado a

def convolucion(im,k,sig):
    iC=cv2.GaussianBlur(im, (k , k), sig)
    #pintaI(iC)
    return iC

def ej1A():
    img1=convolucion(img,5,1)
    img2=convolucion(img,5,2)
    img3=convolucion(img,5,3)

    concatenate((img,img1,img2,img3))


def ej1B(img,dx,dy): #%img imagen, dx,dy=valor de la derivada
    blur=cv2.blur(img,(5,5),1)

    kx,ky=cv2.getDerivKernels(dx,dy,3,normalize=False)

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
def ej2A():
    imgBordes=cv2.copyMakeBorder(src=img,top=15,bottom=15,left=15,right=15,borderType=cv2.BORDER_REFLECT)
    imgConv=convolucion(imgBordes,7,2)
    concatenate((imgBordes,imgConv))


def ej2B():
    imgBordes=cv2.copyMakeBorder(src=img,top=15,bottom=15,left=15,right=15,borderType=0)
    ej1B(imgBordes,0,1)


def ej2C():
    imgBordes=cv2.copyMakeBorder(src=img,top=15,bottom=15,left=15,right=15,borderType=0)
    ej1B(imgBordes,2,0)


def ej2D():
    img=leeimagen("data/cat.bmp",0)

    imgN=cv2.copyMakeBorder(src=img,top=15,bottom=15,left=15,right=15,borderType=4)
    #pintaI(imgN)

    imgPD1=cv2.pyrDown(imgN);
    #pintaI(imgPD1)

    imgPD2=cv2.pyrDown(imgPD1);
    #pintaI(imgPD2)

    imgPD3=cv2.pyrDown(imgPD2);
    #pintaI(imgPD3)

    imgPU1=cv2.pyrUp(imgPD3);
    #pintaI(imgPU1)

    imgPU2=cv2.pyrUp(imgPU1);
    #pintaI(imgPU2)

    imgPU3=cv2.pyrUp(imgPU2);
    #pintaI(imgPU3)



    images=(imgN,imgPD1,imgPD2,imgPD3,imgPD3,imgPU1,imgPU2,imgPU3)
    pintaMI(images)



    """
    o1=concatenateDifSizes(images1)
    o2=concatenateDifSizes(images2)

    cv2.imwrite("o1.jpg", o1)
    cv2.imwrite("o2.jpg", o2)

    o1=leeimagen("o1.jpg",1)
    o2=leeimagen("o2.jpg",1)

    pintaI(o1)
    pintaI(o2)
    """

ej2D()





#

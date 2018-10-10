import cv2
import numpy as np

MAX_KERNEL_LENGTH=11




def concatenateDifSizes(images,p):
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images)
    output=np.zeros((height,width))
    if(p==1):
        output=output+255

    x=0

    for i in images:
        h,w=i.shape
        output[:h,x:x+w,]=i
        x+=w

    output=np.array(output,float)/float(255)
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

#Ejercicio 1
#apartado a

def convolucion(im,k,sig):
    iC = cv2.GaussianBlur(im, (k , k), sig)
    #pintaI(iC)
    return iC

def ej1A(img):
    img1 = convolucion(img,5,1)
    img2 = convolucion(img,5,2)
    img3 = convolucion(img,5,3)

    concatenate((img,img1,img2,img3))


def ej1B(img): #%img imagen, dx,dy=valor de la derivada

    ksize=3

    kx0,ky0=cv2.getDerivKernels(1,0,ksize,normalize=False)
    kx1,ky1=cv2.getDerivKernels(0,1,ksize,normalize=False)


    img1=cv2.sepFilter2D(img,-1,kx0,ky0,delta=100)
    img2=cv2.sepFilter2D(img,-1,kx1,ky1,delta=100)

    concatenate((img1,img2))



def ej1C(img):
    deltaParam=50

    img1=cv2.copyMakeBorder(src=img,top=10,bottom=10,left=10,right=10,borderType=1)
    img2=cv2.copyMakeBorder(src=img,top=10,bottom=10,left=10,right=10,borderType=4)

    simga1=1
    simga3=3
    blur11=cv2.blur(img1,(5,5),simga1)
    blur12=cv2.blur(img2,(5,5),simga1)

    blur31=cv2.blur(img1,(5,5),simga3)
    blur32=cv2.blur(img2,(5,5),simga3)

    imgOut1=cv2.Laplacian(blur11,ksize=3,ddepth=0,delta=deltaParam)
    imgOut2=cv2.Laplacian(blur12,ksize=3,ddepth=0,delta=deltaParam)
    imgOut3=cv2.Laplacian(blur31,ksize=3,ddepth=0,delta=deltaParam)
    imgOut4=cv2.Laplacian(blur32,ksize=3,ddepth=0,delta=deltaParam)

    concatenate((imgOut1,imgOut2))
    concatenate((imgOut3,imgOut4))



#ej2
def ej2A(img,kx,ky):
    imgBordes=cv2.copyMakeBorder(src=img,top=15,bottom=15,left=15,right=15,borderType=cv2.BORDER_REFLECT)
    imgConv=cv2.sepFilter2D(imgBordes,-1,kx,ky,delta=100)
    concatenate((imgBordes,imgConv))


def ej2BC(img,dx,dy):
    img=convolucion(img,3,1)
    kx,ky=cv2.getDerivKernels(dx,dy,3,normalize=False)
    ej2A(img,kx,ky)





def ej2D(img):
    img=leeimagen("data/cat.bmp",0)

    imgN=cv2.copyMakeBorder(src=img,top=15,bottom=15,left=15,right=15,borderType=4)

    imgPD1=cv2.pyrDown(imgN)
    imgPD2=cv2.pyrDown(imgPD1)
    imgPD3=cv2.pyrDown(imgPD2)

    i=concatenateDifSizes((imgN,imgPD1,imgPD2,imgPD3),0)
    pintaI(i)

    return imgN,imgPD3

def ej2E(imgN,imgPD):
    imgPU1=cv2.pyrUp(imgPD)
    imgPU2=cv2.pyrUp(imgPU1)
    imgPU3=cv2.pyrUp(imgPU2)

    delta=40

    imgN= cv2.resize(imgN,(imgPU1.shape[1],imgPU1.shape[0]))
    laplace1=cv2.subtract(imgN,imgPU1)
    laplace1=cv2.add(laplace1,delta)

    imgN= cv2.resize(imgN,(imgPU2.shape[1],imgPU2.shape[0]))
    laplace2=cv2.subtract(imgN,imgPU2)
    laplace2=cv2.add(laplace2,delta)

    imgN= cv2.resize(imgN,(imgPU3.shape[1],imgPU3.shape[0]))
    laplace3=cv2.subtract(imgN,imgPU3)
    laplace3=cv2.add(laplace3,delta)

    i=concatenateDifSizes((imgPD,laplace1,laplace2,laplace3),1)
    pintaI(i)








def main():
    img=leeimagen("data/cat.bmp",0)
    """

    print("Ejercico 1A")
    ej1A(img)

    print("Ejercico 1B")
    ej1B(img)

    print("Ejercico 1C")
    ej1C(img)

    print("Ejercico 2A")
    kx,ky=cv2.getDerivKernels(1,0,3,normalize=False)
    ej2A(img,kx,ky)

    print("Ejercico 2B")
    ej2BC(img,1,1)

    print("Ejercico 2C")
    ej2BC(img,2,2)
    """
    print("Ejercico 2D")
    imgB,imgP=ej2D(img)

    print("Ejercico 2E")
    ej2E(imgB,imgP)

main()

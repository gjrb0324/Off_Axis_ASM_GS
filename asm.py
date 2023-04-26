from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import cv2
import numpy as np
import math
import cmath
from math import *
import os
from multiprocessing import Pool

#Constants
lam = 532*pow(10,-6) #mm scale
pixel = 2.4*pow(10,-3)* 2048
pixel_f = 1/pixel
mag = 10*200/180
na = 0.25
k0 = 2*math.pi/lam
zr = na/(lam*pixel_f*mag)
refractive_index=1.355
im_len=2048
cen_x=0
cen_y=0

#Filter Function
def filtering(index):
    j,i = divmod(index,2048)
    if ((i-cen_x)*(i-cen_x) +(j-cen_y)*(j-cen_y)) <(zr*zr):
        return 1
    else:
        return 0
#Image, FFT
def prop(index):
    global pixel_f
    global im_len
    global lam
    global mag
    j,i = divmod(index,2048)
    i-=2048/2
    j-=2048/2
    alpha = j*pixel_f*lam
    beta = i*pixel_f*lam#*2/3
    sqrt = math.sqrt(1-alpha*alpha-beta*beta)
    prop = cmath.exp(2j*math.pi/lam*sqrt*mag*mag*0.1)
    return prop
def FFT(im):
    img = cv2.imread(im,cv2.IMREAD_GRAYSCALE)
    img_height, img_width = np.shape(img)
    global im_len
    im_len= (img_width if img_width<img_height else img_height)
    img=img[0:im_len:,0:im_len]

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fftd = 20*np.log(np.abs(fshift))

    center = []
    cut = int(math.ceil(zr)+im_len/2)
    temp_len = min(im_len,cut+2*ceil(zr))
    fftd_temp = fftd[cut:2048,cut:2048]
    center_y,center_x = divmod(np.argmax(fftd_temp),2048-cut)
    global cen_y
    cen_y= center_y + cut
    global cen_x
    cen_x= center_x + cut
    center.append((cen_x,cen_y))

    with Pool() as pool:
        filtered = pool.map(filtering,range((im_len)*(im_len)))
    filtered=np.reshape(filtered,(im_len,im_len))
    fshift = fshift*filtered
    ifftd = np.roll(fshift,(-(cen_x-1024),-(cen_y-1024)),axis=(1,0))
    ifftd = np.fft.ifftshift(ifftd)
    #ifftd[2043:2045,2044:2046] = ifftd[2024:2026,2024:2026]
    ifftd = np.fft.ifft2(ifftd)

    return img,f,  fftd, ifftd, center
def AS(recon_field, prop_len):
    #padded = np.pad(recon_field,((512,512),(512,512)))
    f = np.fft.fft2(recon_field)
    fshift = np.fft.fftshift(f)
    with Pool() as pool:
        prop_matrix = pool.map(prop,range((2048*2048)))
    prop_matrix=np.reshape(prop_matrix,(2048,2048))
    fshift = np.multiply(prop_matrix,fshift)
    fftd = 20*np.log(np.abs(fshift))

    ifftd = np.fft.ifftshift(fshift)
    recon = np.fft.ifft2(ifftd)

    return recon

def plotting(img, fftd, recon, centers, zr):
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.set_title('Original Image')
    #img = img[1200:1261,1510:1571]
    ax1.imshow(img,cmap='gray')

    ax2.set_title('Fourier Transformed')
    ax2.imshow(fftd)
    circle1 = plt.Circle((int(im_len/2),int(im_len/2)),2*zr,fill=False)
    circle2 = plt.Circle(centers[0],zr, fill=False)

    ax2.add_artist(circle1)
    ax2.add_artist(circle2)


    ax3.set_title('Reconsturcted- Amplitude')
    amplitude_recon = abs(recon)
    amplitude_recon = amplitude_recon#[1140:1200,740:780]
    ax3.imshow(amplitude_recon,cmap='gray')


    ax4.set_title('Reconstructed- Phase')
    angle = np.angle(recon)
    thickness = angle/((refractive_index-1)*k0)
    thickness = thickness#[1140:1200,740:780]
    ax4.imshow(thickness)

    plt.show()

def main():
    oa_im = 'Data/ASM/0.bmp'
    as_im = 'Data/ASM/100.bmp'
    fig, ((ax0,ax1,ax2),(ax3,ax4,ax5))= plt.subplots(2,3)
    img, f, fftd, recon, centers= FFT(oa_im)
    ax0.set_title('Intensity Recon-Focal')
    recon_amp = abs(recon)
    ax0.imshow(recon_amp[1300:1800,900:1400],cmap='gray')

    ax3.set_title('Phase Recon-Focal')
    angle = np.angle(recon)
    thickness = angle/((refractive_index-1)*k0)
    thickness= thickness
    ax3.imshow(thickness[1300:1800,900:1400])

    img, f, fftd, recon, centers = FFT(as_im)
    ax1.set_title('FFT-Intensity-200um Front')
    recon_amp = abs(recon)
    ax1.imshow(recon_amp[1300:1800,900:1400],cmap='gray')

    ax4.set_title('FFT-phase-200um Front')
    angle = np.angle(recon)
    thickness = angle/((refractive_index-1)*k0)
    thickness= thickness
    ax4.imshow(thickness[1300:1800,900:1400])

    as_recon= AS(recon,2)
    ax2.set_title('Intensity Recon- 200um Front')
    as_recon_amp = abs(as_recon)#[512:im_len+512,512:im_len+512]
    ax2.imshow(as_recon_amp[1300:1800,900:1400],cmap='gray')

    ax5.set_title('Phase Recon-200um Front')
    as_angle = np.angle(as_recon)
    as_thickness = as_angle/((refractive_index-1)*k0)
    as_thickness = as_thickness#[512:im_len+512,512:im_len+512]
    ax5.imshow(as_thickness[1300:1800,900:1400])




    plt.show()


if __name__ == '__main__':
    main()


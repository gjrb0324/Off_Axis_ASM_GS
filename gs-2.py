import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

im_len=0
prop_len=0
lam = 532*pow(10,-6)
pixel = 2.4*pow(10,-3)*2048
pixel_f = 1/pixel
mag = 10*200/180
k0= 2*math.pi/lam
refractive_index = 1.355
na =0.25
zr = na/(lam*pixel_f*mag)

def filtering(index):
    j,i = divmod(index,2048)
    if ((i-cen_x)*(i-cen_x) +(j-cen_y)*(j-cen_y)) <(zr*zr):
        return 1
    else:
        return 0

def OFFAXIS(im):
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
    temp_len = min(im_len,cut+2*math.ceil(zr))
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
    ifftd = np.fft.ifft2(ifftd)

    return ifftd

def prop(index):
    global pixel_f
    global im_len
    global lam
    global mag
    j,i = divmod(index,512)
    i-=512/2
    j-=512/2
    alpha = j*pixel_f*lam
    beta = i*pixel_f*lam
    sqrt = math.sqrt(1-alpha*alpha-beta*beta)
    prop = np.exp(2j*math.pi/lam*sqrt*mag*mag*prop_len)
    return prop

def AS(field, propa_len):
    global prop_len
    prop_len = propa_len
    fftd = np.fft.fft2(field)
    fshift = np.fft.fftshift(fftd)
    with Pool() as pool:
        prop_matrix = pool.map(prop,range((512*512)))
    prop_matrix=np.reshape(prop_matrix,(512,512))
    fshift = np.multiply(prop_matrix,fshift)
    recon = np.fft.ifftshift(fshift)
    recon = np.fft.ifft2(recon)
    return recon



def GS(img_0,img_1):
    global im_len
    src_amp = np.sqrt(img_0)
    trg_amp=np.sqrt(img_1)
    loss = []
    pbar = tqdm(range(0,30))
    wave_e = AS(trg_amp,0.1)
    for i in pbar:
        wave = src_amp*np.exp(1j*np.angle(wave_e))
        wave_pe = AS(wave,-0.1)
        wave_p = trg_amp*np.exp(1j*np.angle(wave_pe))
        wave_e  = AS(wave,0.1)
        ang = np.angle(wave_e)
        thickness = ang/(0.355*k0)
        plt.imshow(thickness)
        plt.pause(0.5)
    plt.show()

    src_phs = np.angle(wave_e)

    return src_amp, src_phs




def main():
    ref = 'Data/GS/Reference.bmp'
    recon = OFFAXIS(ref)
    thickness = np.angle(recon)/(0.355*k0)
    thickness = thickness[1300:1812,850:1362]
    plt.imshow(thickness[100:200,400:500])
    plt.savefig('OFFAXIS.')
    img_0='Data/GS/0.bmp'
    img_1='Data/GS/100.bmp'
    im_0 = cv2.imread(img_0,cv2.IMREAD_GRAYSCALE)
    im_1 = cv2.imread(img_1,cv2.IMREAD_GRAYSCALE)
    img_height, img_width = np.shape(im_0)
    global im_len
    im_len= min(img_height,img_width)
    img_0=im_0[1300:1812,850:1362]
    img_1=im_1[1300:1812,850:1362]
    src_amp, src_phs = GS(img_0,img_1)

    thickness = src_phs/(0.355*k0)
    plt.set_title('Phase Reconstructed')
    plt.imshow(thickness)
    plt.show()


if __name__ == '__main__':
    main()

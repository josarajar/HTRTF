#!/usr/bin/env python3

"""
    `Modules implementing affine and morphological distortions to images in
    numpy datatype.
"""
from PIL import Image
import numpy as np


def scale(img, scale_prob = 0.5, scale_stdv = 0.12):
    """
    It scale the image provided to a random size chosen from the sample of a lognormal probability
    function with scale_stdv standard deviation. The probabilite that the image will be scaled is scale_prob.
    
    Args:
        img: numpy format image.
        scale_prob: probability of scaling the image.
        scale_stdv: standard deviation of the lognormal probability function used to chose
            the size of the scaled image.
            
    Returns:
        img: the scaled (or not) image.
    
    """
    scale = np.random.binomial(1, scale_prob)
    
    if scale:
        
        imgPIL = Image.fromarray(img)
        ho, vo = imgPIL.size
        scale_factor = np.random.lognormal(sigma=scale_stdv)
        hn, vn = int(scale_factor*ho), int(scale_factor*vo)
        img_sc = imgPIL.resize((hn, vn))
        
        img = np.array(img_sc).reshape((vn,hn))
                
        if hn > ho:
            img = img[int(vn/2)-int(vo/2):int(vn/2)+int(np.ceil(vo/2)) ,int(hn/2)-int(ho/2):int(hn/2)+int(np.ceil(ho/2))] 
        else:
            img = np.pad(img, ((int((vo-vn)/2), int(np.ceil((vo-vn)/2))),((int((ho-hn)/2), int(np.ceil((ho-hn)/2))))), mode='constant')
            
    return img

def shear(img, shear_prob = 0.5, shear_prec = 4):
    """
    It shear the image provided to a random angle chosen from the sample of a vonmises probability
    function with shear_prec kappa parameter. The probability that the image will be sheared is shear_prob.
    
    Args:
        img: numpy format image.
        shear_prob: probability of shearing the image.
        shear_prec: kappa parameter of the vonmises probability function used to chose
            the angle to shear image.
            
    Returns:
        img: the sheared (or not) image.
    
    """
    shear = np.random.binomial(1, shear_prob)

    if shear:
        import cv2
        rows,cols = img.shape
        shear_angle = np.random.vonmises(0, kappa = shear_prec)
        m = np.tan(shear_angle)
        
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[50,50],[200,50],[50+m*150,200]])
        M = cv2.getAffineTransform(pts1,pts2)
            
        img = cv2.warpAffine(img,M,(cols,rows))
        
    return img

def rotate(img, rotate_prob = 0.5, rotate_prec = 100):
    """
    It rotate the image provided to a random angle chosen from the sample of a vonmises probability
    function with rotate_prec kappa parameter. The probability that the image will be sheared is rotate_prob.
    
    Args:
        img: numpy format image.
        rotate_prob: probability of rotating the image.
        rotate_prec: kappa parameter of the vonmises probability function used to chose
            the angle to rotate image.
            
    Returns:
        img: the rotated (or not) image.
    
    """    
    rotate = np.random.binomial(1, rotate_prob)
    
    if rotate:
        import cv2
        rows,cols = img.shape
        rotate_prec = rotate_prec * max(rows/cols, cols/rows)
        rotate_angle = np.random.vonmises(0, kappa = rotate_prec)*180/np.pi
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_angle,1)
        img = cv2.warpAffine(img,M,(cols,rows))
    
    return img

def translate(img, translate_prob = 0.5, translate_stdv = 0.02):
    """
    It translate the image provided to a random position chosen from the sample of a normal probability
    function with translate_stdv standard deviation. The probability that the image will be translated is tanslate_prob.
    
    Args:
        img: numpy format image.
        translate_prob: probability of translating the image.
        translate_stdv: standard deviation of the normal probability function used to chose
            the position to translate the image.
            
    Returns:
        img: the translated (or not) image.
    
    """
    translate = np.random.binomial(1, translate_prob)
    
    if translate:
        import cv2
        rows,cols = img.shape
        h_translation_factor = np.random.normal(0, scale = translate_stdv * cols)
        v_translation_factor = np.random.normal(0, scale = translate_stdv * rows)
        M = np.float32([[1,0,h_translation_factor],[0,1,v_translation_factor]])
        img = cv2.warpAffine(img,M,(cols,rows))
    
    return img

def dilate(img, dilation_prob = 0.5 , dilation_srate = 0.4, dilation_rrate = 1):
    
    dilate = np.random.binomial(1, dilation_prob)
    
    if dilate:
        import cv2
        kernel_size = np.min([2*np.random.geometric(dilation_srate)+1, 15])
        kernel = np.zeros([kernel_size, kernel_size])
        center = np.array([int(kernel_size/2), int(kernel_size/2)])
        for x in range(kernel_size):
            for y in range(kernel_size):
                d = np.linalg.norm(np.array([x,y])-center)
                p = np.exp(-d*1)
                value = np.random.binomial(1, p)
                kernel[x,y] = value or 10**-16
        
        img = cv2.dilate(img,kernel,iterations = 1)
    
    return img

def erode(img, erosion_prob = 0.5 , erosion_srate = 0.8, erosion_rrate = 1.2):
    
    erode = np.random.binomial(1, erosion_prob)
    
    if erode:
        import cv2
        kernel_size = np.min([2*np.random.geometric(erosion_srate)+1, 15])
        kernel = np.zeros([kernel_size, kernel_size])
        center = np.array([int(kernel_size/2), int(kernel_size/2)])
        for x in range(kernel_size):
            for y in range(kernel_size):
                d = np.linalg.norm(np.array([x,y])-center)
                p = np.exp(-d*1)
                value = np.random.binomial(1, p)
                kernel[x,y] = value or 10**-16
        
        img = cv2.erode(img,kernel,iterations = 1)
    
    return img

def distort(img_list):
    new_list=[]
    for ind, img_np in enumerate(img_list):
        #img_np = 255 - img_np
        img_np = translate(img_np)
        img_np = rotate(img_np)
        img_np = shear(img_np)
        img_np = scale(img_np)
        img_np = dilate(img_np)
        img_np = erode(img_np)
        new_list.append(img_np)
    
    return new_list
    
def main(): 
    img = Image.open('Path to the image')
    img_npo = np.array(img).reshape(img.size[1], img.size[0])
    
    print(img.size)
    
    for ind in range(10):
        img_np = 255 - img_npo
        img_np = translate(img_np)
        img_np = rotate(img_np)
        img_np = shear(img_np)
        img_np = scale(img_np)
        img_np = dilate(img_np)
        img_np = erode(img_np)
        
    #    img_sc = Image.fromarray(img_dist)
    #    print(img_sc.size)
    #    
    #    img_sc.show()
    #    img_sh = erode(img_np, erosion_prob = 1)
        img = Image.fromarray(img_np)
        print(img.size)
        img.show()



if __name__ == '__main__':
    main()

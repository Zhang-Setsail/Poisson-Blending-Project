import numpy as np
import cv2
import math

import numpy as np
from scipy import fftpack,signal

import cgd

def possion_copy(dest_img_name, copy_img_name, copy_mask_name):
    dest_img = cv2.imread(dest_img_name)
    copy_img = cv2.imread(copy_img_name)
    copy_mask = cv2.imread(copy_mask_name)

    origin_divergence = cv2.Laplacian(dest_img, cv2.CV_64F)
    copy_divergence = cv2.Laplacian(copy_img, cv2.CV_64F)

    copy_mask = copy_mask.astype(np.float64)
    
    copy_mask_single = copy_mask[:,:,0]
    copy_mask_single = np.where(copy_mask_single>1, 1, copy_mask_single)
    target_divergence = np.zeros(origin_divergence.shape)
    print(origin_divergence[:,:,0].shape)
    print((np.ones(copy_mask_single.shape)-copy_mask_single).shape)
    target_divergence[:,:,0] = cv2.multiply(origin_divergence[:,:,0], np.ones(copy_mask_single.shape)-copy_mask_single) + cv2.multiply(copy_divergence[:,:,0], copy_mask_single)
    # target_divergence[:,:,0] = cv2.multiply(origin_divergence[:,:,0], np.ones(copy_mask_single.shape)-copy_mask_single)
    target_divergence[:,:,1] = cv2.multiply(origin_divergence[:,:,1], np.ones(copy_mask_single.shape)-copy_mask_single) + cv2.multiply(copy_divergence[:,:,1], copy_mask_single)
    # target_divergence[:,:,1] = cv2.multiply(origin_divergence[:,:,1], np.ones(copy_mask_single.shape)-copy_mask_single)
    target_divergence[:,:,2] = cv2.multiply(origin_divergence[:,:,2], np.ones(copy_mask_single.shape)-copy_mask_single) + cv2.multiply(copy_divergence[:,:,2], copy_mask_single)
    # target_divergence[:,:,2] = cv2.multiply(origin_divergence[:,:,2], np.ones(copy_mask_single.shape)-copy_mask_single)

    return target_divergence

def main():
    '''
    dest_img:是原图像
    copy_img:是想要复制的图像
    copy_mask:是想要使用的mask
    '''
    dest_img_path = './input/1/target.png'
    copy_img_path = './input/1/source.png'
    copy_mask_path = './input/1/mask.png'
    dest_img = cv2.imread('./input/1/target.png')
    copy_img = cv2.imread('./input/1/source.png')
    copy_mask = cv2.imread('./input/1/mask.png')

    origin_divergence = cv2.Laplacian(dest_img, cv2.CV_64F)
    # print(dest_img.shape)
    # print(copy_img.shape)
    # print(copy_mask.shape)
    # print(np.max(copy_mask))
    # copy_mask = np.where(copy_mask>1, 1, copy_mask)

    copy_mask_single = copy_mask[:,:,0]
    copy_mask_single = np.where(copy_mask_single>1, 1, copy_mask_single)
    # count = 0
    # for i in range(400):
    #     for j in range(300):
    #         if(copy_mask[i,j,0]>1):
    #             count = count + 1
    # print(count)
    # print(np.max(copy_mask))

    target_divergence = possion_copy(dest_img_path, copy_img_path, copy_mask_path)

    integrated_image = cgd.conjugate_gradient_descent(target_divergence, dest_img, copy_mask_single)

    cv2.imwrite('target_divergence.jpg', cv2.convertScaleAbs(target_divergence))
    cv2.imwrite('dest_divergence.jpg', cv2.convertScaleAbs(cv2.Laplacian(dest_img, cv2.CV_64F)))
    cv2.imwrite('copy_divergence.jpg', cv2.convertScaleAbs(cv2.Laplacian(copy_img, cv2.CV_64F)))
    cv2.imwrite('final_result.jpg', integrated_image)

main()
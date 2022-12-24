import numpy as np
import cv2
import math

def conjugate_gradient_descent(target_divergence, boundary_values, image_mask):
    # target_divergence, initialization, boundary_mask, boundary_values, convergence_parameter, iteration
    # boundary_values是原图
    # target_divergence目标梯度图
    # initialization是全0图
    # convergence_parameter是error tolerance最大是1
    # iteration是最大循环数

    # image mask 是一个遮罩-标识出需要copy的部分-需要:copy部分value是1,unchanged的部分value是0
    iteration = 50000
    convergence_parameter = 0.9
    initialization_3 = np.zeros(boundary_values.shape)
    initialization = initialization_3[:,:,0]
    boundary_mask_3 = np.ones(boundary_values.shape)
    boundary_mask = boundary_mask_3[:,:,0]

    one_array = boundary_mask.copy()
    boundary_mask[0,:] = 0
    boundary_mask[:,0] = 0
    boundary_mask[-1,:] = 0
    boundary_mask[:,-1] = 0

    boundary_mask = image_mask
    boundary_mask = boundary_mask.astype(np.float64)
    one_array = np.ones(boundary_mask.shape)
    # print(boundary_mask)

    boundary_values_r = boundary_values[:,:,0]
    boundary_values_g = boundary_values[:,:,1]
    boundary_values_b = boundary_values[:,:,2]
    target_divergence_r = target_divergence[:,:,0]
    target_divergence_g = target_divergence[:,:,1]
    target_divergence_b = target_divergence[:,:,2]

    # integrated_image是最后输出
    integrated_image = np.zeros(boundary_values.shape)
    # r
    integrated_image_r = integrated_image[:,:,0]
    boundary_values_r = boundary_values_r.astype(np.float64)
    print(boundary_mask.shape)
    print(initialization.shape)
    print(boundary_mask.dtype)
    print(initialization.dtype)
    cv2.multiply(boundary_mask,initialization)
    integrated_image_r = cv2.multiply(boundary_mask,initialization) + cv2.multiply((one_array - boundary_mask), boundary_values_r)
    r = cv2.multiply(boundary_mask, target_divergence_r-cv2.Laplacian(integrated_image_r, cv2.CV_64F))
    # print(np.max(target_divergence))
    # print(np.min(target_divergence))
    d = r
    # print(r.shape)
    # print(r.T.shape)
    new_error = np.trace(np.dot(r.T,r))
    n = 0
    while (math.sqrt(new_error)>convergence_parameter and n<iteration):
        q = cv2.Laplacian(d, cv2.CV_64F)
        eta = new_error / np.trace(np.dot(d.T,q))
        integrated_image_r = integrated_image_r + cv2.multiply(boundary_mask, eta * d)
        r = cv2.multiply(boundary_mask, r-eta*q)
        old_error = new_error
        new_error = np.trace(np.dot(r.T,r))
        beta = new_error / old_error
        d = r + beta * d
        n = n + 1
        print(new_error)
        print(n)
    
    # g
    integrated_image_g = integrated_image[:,:,1]
    boundary_values_g = boundary_values_g.astype(np.float64)
    integrated_image_g = cv2.multiply(boundary_mask,initialization) + cv2.multiply((one_array - boundary_mask), boundary_values_g)
    g = cv2.multiply(boundary_mask, target_divergence_g-cv2.Laplacian(integrated_image_g, cv2.CV_64F))
    d = g
    new_error = np.trace(np.dot(g.T,g))
    n = 0
    while (math.sqrt(new_error)>convergence_parameter and n<iteration):
        q = cv2.Laplacian(d, cv2.CV_64F)
        eta = new_error / np.trace(np.dot(d.T,q))
        integrated_image_g = integrated_image_g + cv2.multiply(boundary_mask, eta * d)
        g = cv2.multiply(boundary_mask, g-eta*q)
        old_error = new_error
        new_error = np.trace(np.dot(g.T,g))
        beta = new_error / old_error
        d = g + beta * d
        n = n + 1
        print(new_error)
        print(n)
    
    # b
    integrated_image_b = integrated_image[:,:,2]
    boundary_values_b = boundary_values_b.astype(np.float64)
    integrated_image_b = cv2.multiply(boundary_mask,initialization) + cv2.multiply((one_array - boundary_mask), boundary_values_b)
    b = cv2.multiply(boundary_mask, target_divergence_b-cv2.Laplacian(integrated_image_b, cv2.CV_64F))
    d = b
    new_error = np.trace(np.dot(b.T,b))
    n = 0
    while (math.sqrt(new_error)>convergence_parameter and n<iteration):
        q = cv2.Laplacian(d, cv2.CV_64F)
        eta = new_error / np.trace(np.dot(d.T,q))
        integrated_image_b = integrated_image_b + cv2.multiply(boundary_mask, eta * d)
        b = cv2.multiply(boundary_mask, b-eta*q)
        old_error = new_error
        new_error = np.trace(np.dot(b.T,b))
        beta = new_error / old_error
        d = b + beta * d
        n = n + 1
        print(new_error)
        print(n)


    integrated_image[:,:,0] = integrated_image_r
    integrated_image[:,:,1] = integrated_image_g
    integrated_image[:,:,2] = integrated_image_b
    # print(np.max(integrated_image_r))
    # print(np.max(integrated_image_g))
    # print(np.max(integrated_image_b))

    # print(integrated_image_r)
    # cv2.imwrite('final_result_r.jpg', integrated_image_r)
    # cv2.imwrite('final_result_g.jpg', integrated_image_g)
    # cv2.imwrite('final_result_b.jpg', integrated_image_b)
    # cv2.imwrite('final.jpg', np.abs(integrated_image))
    # print(integrated_image.dtype)
    # print(integrated_image.shape)
    # print(np.abs(integrated_image.astype(np.int8))/255)

    # print(np.min(np.abs(integrated_image.astype(np.int8))/255))
    # print(np.max(np.abs(integrated_image.astype(np.int8))/255))
    # print((np.abs(integrated_image.astype(np.int8))/255).dtype)
    # test = (np.abs(integrated_image.astype(np.int8))/255 + 0.5).clip(0,1)
    # test2 = (np.abs(integrated_image.astype(np.int8))/255).clip(0,1)
    # cv2.imwrite('final_int.jpg', np.abs(integrated_image.astype(np.int8))/255)
    # cv2.imwrite('test.jpg', test)
    # cv2.imwrite('test2.jpg', test2)
    # test3 = (test * 255).astype(np.int8)
    # print(np.abs(test3))
    # cv2.imwrite('test3.jpg', np.abs(test3))


    return integrated_image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# for img in os.listdir("./img/cats/8"):
#     image = cv2.imread(f"./img/cats/8/{img}")[... ,::-1]/255.0
#     noise = np.random.normal(loc=0, scale=1, size=image.shape)
#     noisy = np.clip((image + noise*0.2),0, 1)
#     # cv2.imshow("v", noisy)
#     # k = cv2.waitKey(1)
#     result = cv2.normalize(noisy, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     cv2.imwrite(f'./img/cats/9/{img}', result)


image = cv2.imread(f"./img/other_cats/photo/cat.jpg")[... ,::-1]/255.0
noise = np.random.normal(loc=0, scale=1, size=image.shape)
noisy = np.clip((image + noise*0.2),0, 1)
# cv2.imshow("v", noisy)
# k = cv2.waitKey(1)
result = cv2.normalize(noisy, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite(f'./img/other_cats/photo/cat.jpg', result)
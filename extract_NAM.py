import cv2
import scipy.io as sio
import numpy as np
import os
data_path = '[YourPath]/NAM/'

tem = data_path+'Nikon_D600/'+'ISO_3200/'+'C_1.mat'
name = tem.split('/')
name[-1] = name[-1].split('.')[0]
name = '_'.join(name[-3:])

save_path = './NAM/'
gt_path = save_path+'GT/'
ny_path = save_path +'NY/'

if not os.path.exists(gt_path):
    os.mkdir(gt_path)

if not os.path.exists(ny_path):
    os.mkdir(ny_path)

print(name)
file = sio.loadmat(tem)
print(file.keys())
# print(file['img_mean'])
# print(file['img_noisy'].shape)
img_mean = np.array(file['img_mean'])
img_noisy = np.array(file['img_noisy'])

cv2.imwrite(gt_path+'%s_mean.PNG'%name, cv2.cvtColor(img_mean, cv2.COLOR_RGB2BGR))
cv2.imwrite(ny_path+'%s_noisy.PNG'%name, cv2.cvtColor(img_noisy, cv2.COLOR_RGB2BGR))
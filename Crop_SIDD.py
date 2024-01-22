import os
import cv2
import glob
import numpy as np

import os
import fnmatch


cropSize = 512
num = 100

path = '[YourPath]/SSID/SIDD_Medium_Srgb/Data/'
assert os.path.exists(path) #用来判断路径是否存在

GT = []
Noisy = []

for root,dirs,files in os.walk(path):
    for name in files:
        if fnmatch.fnmatch(name,'*GT*'):
            GT.append(root+'/'+name)
        if fnmatch.fnmatch(name,'*NOISY*'):
            Noisy.append(root+'/'+name)
# Crop SIDD

GT.sort()
Noisy.sort()
print(len(GT),len(Noisy))

out_dir =  './datasets/SIDD_Medium_Srgb_Patches_512'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(os.path.join(out_dir, 'GT')):
    os.mkdir(os.path.join(out_dir, 'GT'))
if not os.path.exists(os.path.join(out_dir, 'Noisy')):
    os.mkdir(os.path.join(out_dir, 'Noisy'))

# 不改成多线程还是有点点慢

for i in range(len(GT)):
    image = cv2.imread(GT[i])
    noisy_image = cv2.imread(Noisy[i])
    endw, endh = image.shape[0], image.shape[1]
    assert (endw >= cropSize) and (endh >= cropSize)
    for k in range(num):
        x = np.random.randint(0, endw - cropSize)
        y = np.random.randint(0, endh - cropSize)
        crop = image[x:(cropSize + x), y:(cropSize + y), :]
        noisy_crop = noisy_image[x:(cropSize + x), y:(cropSize + y), :]
        cv2.imwrite(os.path.join(out_dir, 'GT', '%d_%d.PNG'%(i, k)), crop)
        cv2.imwrite(os.path.join(out_dir, 'Noisy', '%d_%d.PNG' % (i, k)), noisy_crop)
import scipy
import os 
import numpy as np
from codes.data.util import read_img_array
import scipy.io as sio
import cv2

files = sio.loadmat(os.path.join('[YourPath]/SSID/test', 'BenchmarkNoisyBlocksSrgb.mat'))
imgArray = files['BenchmarkNoisyBlocksSrgb'] # shape (40,32,256,256,3)
nImages = 40
nBlocks = imgArray.shape[1]
DenoisedBlocksSrgb = np.empty_like(imgArray)
out_dir = os.path.join('./SSID/test/', 'img')
print(out_dir)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

print(imgArray.shape)
for i in range(nImages):
    Inoisy = read_img_array(imgArray[i])

    for k in range(nBlocks):
        img = Inoisy[k]
        img = (img * 255.0).round()
        save_file = os.path.join(out_dir, '%d_%02d.PNG' % (i , k))
        cv2.imwrite(save_file,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print('[%d/%d] is done\n' % (i+1, 40))

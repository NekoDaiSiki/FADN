import h5py
import os
import numpy as np
import cv2
# save image from matlile
sourcepath = '[YourPath]/DND/'
out_path = './DND/DND_image/'

infos = h5py.File(sourcepath+'info.mat','r')
info = infos['info']
bb = info['boundingboxes']

for i in range(50):
     filename = os.path.join(sourcepath, 'images_srgb', '%04d.mat'%(i+1))
     img = h5py.File(filename, 'r')
     Inoisy = np.float32(np.array(img['InoisySRGB']).T)
     ref = bb[0][i]
     boxes = np.array(info[ref]).T
     for k in range(20):
        idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
        Inoisy_crop = Inoisy[idx[0]:idx[1],idx[2]:idx[3],:].copy()
        save_file = os.path.join(out_path, '%04d_%02d.PNG' % (i+1, k+1))
        cv2.imwrite(save_file, cv2.cvtColor(Inoisy_crop*255, cv2.COLOR_RGB2BGR))
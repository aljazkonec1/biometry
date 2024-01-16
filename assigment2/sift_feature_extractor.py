import skimage
import numpy as np
import os
import glob
from skimage import transform, io
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, SIFT, plot_matches
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from skimage import exposure

base_path = os.path.join(os.getcwd() ,'datasets', 'ears', 'images-cropped')
out_dir = os.path.join(base_path, 'test_sift')
input_dir = os.path.join(base_path, 'test')

image_files = glob.glob(os.path.join(input_dir, '*.png'))

descriptor = SIFT()
for image_path in image_files:
    name = os.path.splitext(os.path.basename(image_path))[0]
    print(name)
    img = rgb2gray(skimage.io.imread(image_path))
    img = transform.resize(img, (128, 128))
    img = gaussian(img, 1/3,truncate=2 )
    try :
        descriptor.detect_and_extract(img)
    except:
        equalized_img = exposure.equalize_hist(img)
        descriptor.detect_and_extract(equalized_img)


    descriptors = descriptor.descriptors
    
    np.savetxt(os.path.join(out_dir, name + ".csv"), descriptors, delimiter=",")

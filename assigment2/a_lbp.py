from operator import le
import re
import cv2
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, feature, color, transform
from sklearn.metrics.pairwise import cosine_similarity

class ALBP():


    def ROR(self, x, P):
        shifts = []
        for i in range(P):
            shifts.append(((x >> i) | (x << (P - i))) & ((1 << P) - 1))
        
        return min(shifts)

    def uniform_pattern(slef, val_array ):
        s = 0
        for i in range(1,len(val_array)):
            if val_array[i] != val_array[i-1]:
                s += 1
        s = s + abs(val_array[-1] - val_array[0])

        if s <= 2:
            return sum(val_array)
        else:
            return len(val_array) + 1



    def get_pixel(self,img, center, x, y):
        new_value = 0
        try:
            if img[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value

    def calculate_lbp(self,img, x, y, r, l):
        center = img[x][y]
        val_ar = []

        for i in range(l):
            lbp_pr = self.get_pixel(img, center, x - int(round(r * np.sin(2 * np.pi * i / l))), y+ int(round(r * np.cos(2 * np.pi * i / l)))) #LBP_P,R
            val_ar.append(lbp_pr)
        
        return self.uniform_pattern(val_ar)


    def a_lbp(self,img, r = 1, l = 8):

        lbp = np.zeros_like(img, dtype=np.uint8)
        height, width = img.shape
        for i in range( height-1):
            for j in range( width-1):
                lbp[i, j] = self.calculate_lbp(img, i, j, r, l)

        return lbp

    def generate_a_lbps(self,path, savePath, r = 1, l = 8):
        img_list = []
        for img in glob.glob(path + "/*.png"):
            img_list.append(img)
        
        for img_path in img_list:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (128, 128))
            lbp = self.a_lbp(img, r, l)
            splits = (lbp.reshape(16, 8, -1, 8).swapaxes(1, 2).reshape(-1, 8, 8))
            h = []
            for s in splits:
                h.extend(np.histogram(s)[0])

            np.savetxt(savePath + "/"+os.path.splitext(os.path.basename(img_path))[0] + ".csv", h, delimiter=",")
            # cv2.imwrite(savePath + "/"+os.path.basename(img_path), lbp)

    def skimage_lbp(self,path, savePath, r = 1, l = 8):
        img_list = []
        for img in glob.glob(path + "/*.png"):
            img_list.append(img)
        
        for img_path in img_list:
            # img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (128, 128))
            lbp = feature.local_binary_pattern(img, l, r, method="uniform")
            splits = (lbp.reshape(16, 8, -1, 8).swapaxes(1, 2).reshape(-1, 8, 8))
            h = []
            for s in splits:
                h.extend(np.histogram(s)[0])

            np.savetxt(savePath + "/"+os.path.splitext(os.path.basename(img_path))[0] + ".csv", h, delimiter=",")



    def show_image_with_lbp(self,image, lbp):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        ax = axes.ravel()
        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Original Image')

        ax[1].imshow(lbp, cmap=plt.cm.gray)
        ax[1].set_title('Local Binary Pattern (LBP)')
        
        for a in ax:
            a.axis('off')

        plt.show()

if __name__ == "__main__":
    albp = ALBP()

    wd = os.path.join(os.getcwd(), 'datasets', 'ears', 'images-cropped')
    albp.generate_a_lbps(os.path.join(wd, 'test'), os.path.join(wd, 'test_lbp'))
    albp.skimage_lbp(os.path.join(wd, 'test'), os.path.join(wd, 'test_sk_lbp'))



    # albp.generate_a_lbps(os.getcwd() + "/assigment1/crop_test_gt", os.getcwd() + "/assigment1/crop_test_gt_a_lbp")
    # albp.generate_a_lbps(os.getcwd() + "/assigment1/vj_cropped_test", os.getcwd() + "/assigment1/crop_test_vj_a_lbp")
    # albp.generate_a_lbps(os.getcwd() + "/assigment1/crop_ears_gt", os.getcwd() + "/assigment1/crop_ears_gt_a_lbp")
    # albp.generate_a_lbps(os.getcwd() + "/assigment1/vj_cropped_ears", os.getcwd() + "/assigment1/crop_ears_vj_a_lbp")

    # albp.generate_a_lbps(os.getcwd() + "/assigment1/crop_ears_gt", os.getcwd() + "/assigment1/ears_gt_a_lbp_2_16",2, 16)
    # albp.generate_a_lbps(os.getcwd() + "/assigment1/crop_test_gt", os.getcwd() + "/assigment1/test_gt_a_lbp_2_16", 2, 16)

    # # albp.generate_a_lbps(os.getcwd() + "/assigment1/vj_cropped_ears", os.getcwd() + "/assigment1/ears_vj_a_lbp_2_16", 2, 16)
    # # albp.generate_a_lbps(os.getcwd() + "/assigment1/vj_cropped_test", os.getcwd() + "/assigment1/test_vj_a_lbp_2_16", 2, 16)

    # albp.skimage_lbp(os.getcwd() + "/assigment1/crop_test_gt", os.getcwd() + "/assigment1/test_gt_skimage_lbp_2_16", 2, 16)
    # albp.skimage_lbp(os.getcwd() + "/assigment1/vj_cropped_test", os.getcwd() + "/assigment1/test_vj_skimage_lbp_2_16", 2, 16)
    # albp.skimage_lbp(os.getcwd() + "/assigment1/crop_ears_gt", os.getcwd() + "/assigment1/ears_gt_skimage_lbp_2_16", 2, 16)
    # albp.skimage_lbp(os.getcwd() + "/assigment1/vj_cropped_ears", os.getcwd() + "/assigment1/ears_vj_skimage_lbp_2_16", 2, 16)


    # albp.skimage_lbp(os.getcwd() + "/assigment1/crop_test_gt", os.getcwd() + "/assigment1/crop_test_gt_skimage_lbp")
    # albp.skimage_lbp(os.getcwd() + "/assigment1/vj_cropped_test", os.getcwd() + "/assigment1/crop_test_vj_skimage_lbp")
    # albp.skimage_lbp(os.getcwd() + "/assigment1/crop_ears_gt", os.getcwd() + "/assigment1/crop_ears_gt_skimage_lbp")
    # albp.skimage_lbp(os.getcwd() + "/assigment1/vj_cropped_ears", os.getcwd() + "/assigment1/crop_ears_vj_skimage_lbp")

    # albp.generate_a_lbps(os.getcwd() + "/assigment1/", os.getcwd() + "/assigment1/")
    
    # path = os.getcwd() + "/assigment1"
    # img = cv2.imread(path+ "/crop_test_gt/0501_cropped.png", cv2.COLOR_BGR2GRAY)
    # lbp = a_lbp(img, 1, 8)

    # show_image_with_lbp(img, lbp)

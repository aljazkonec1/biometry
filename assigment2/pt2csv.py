import numpy as np
import os
import pandas as pd
import glob
import torch
import pandas as pd


path = os.path.join(os.getcwd(), 'datasets', 'ears', 'images-cropped', 'test_resnet_pt')

features = glob.glob(os.path.join(path, '*.pt'))
save_path = os.path.join(os.getcwd(), 'datasets', 'ears', 'images-cropped', 'test_resnet')
# save each pt as .csv file
for f_path in features:
    f = torch.load(f_path)  
    f = f.numpy()
    f_name = os.path.basename(f_path).split('.')[0] + '.csv'
    np.savetxt(os.path.join(save_path, f_name), f, delimiter=",")



# USE THIS FILE ONLY FOR THE FINAL EUCILNICA SUBMISSION!

from ultralytics import YOLO
import os, itertools, datetime
import time
import pandas as pd
# Move to the dir of this script        
dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)

# Load the model, feel free to try other models
res = []

model = YOLO("point1_best1.pt")

metrics = model.val(data="ears_final_test.yaml")
m = metrics.results_dict
m["model"] = "point1_best1"
res.append(m)


time.sleep(5)
model = YOLO("point1_best2.pt")
metrics = model.val(data="ears_final_test.yaml")
m = metrics.results_dict
m["model"] = "point1_best2"
res.append(m)


time.sleep(5)
model = YOLO("point2_best1.pt")
metrics = model.val(data="ears_final_test.yaml")
m = metrics.results_dict
m["model"] = "point2_best1"
res.append(m)


time.sleep(5)
model = YOLO("point2_best2.pt")
metrics = model.val(data="ears_final_test.yaml")
m = metrics.results_dict
m["model"] = "point2_best2"
res.append(m)

df = pd.DataFrame(res)

df.to_excel("test_results.xlsx")

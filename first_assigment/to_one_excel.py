import os
import yaml
import pandas as pd

dir = "/home/aljaz/FAKS/biometry/runs/detect"

#open every folder in dir and check if there is a args.yaml and results.csv, open both
res = {}
res_all = []

for folder in os.listdir(dir):
    print(folder)
    args_path = os.path.join(dir, folder, "args.yaml")
    results_path = os.path.join(dir, folder, "results.csv")
    if os.path.isfile(args_path) and os.path.isfile(results_path):
        with open(args_path, 'r') as file:
            args = yaml.safe_load(file)

        results = pd.read_csv(results_path)
        res["folder"] = str(folder)
        res["mAP@50-90"] = results["    metrics/mAP50-95(B)"].iloc[-1] 
        res["lr0"] = args.get("lr0", None)
        res["dropout"] = args.get("dropout", None)
        res["weight_decay"] = args.get("weight_decay", None)
        res["optimizer"] = args.get("optimizer", None)

        res_all.append(res)
        res = {}
df = pd.DataFrame(res_all)
df = df.sort_values(by=['mAP@50-90'], ascending=False)
df.to_excel("results.xlsx")


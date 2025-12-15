import json
import numpy as np

data = np.load("./data/processed/train.npz")
class_names = data["class_names"].tolist()

with open("class_names.json", "w") as f:
    json.dump(class_names, f)
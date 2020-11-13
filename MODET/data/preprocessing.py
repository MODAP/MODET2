import csv
import json
import requests
import numpy as np
from PIL import Image

# {top, left, height, width}

class Dataset():
    def __init__(self, datafile):
        raw_images = []
        raw_labels = []
        with open(datafile, "r") as df:
            reader = csv.reader(df)
            next(reader)
            for i in reader:
                raw_images.append(i[2])
                raw_labels.append(i[3])
        self.labels = [[list(e["bbox"].values()) for e in (json.loads(i))["objects"]] for i in raw_labels]
        self.images = [np.asarray(self.get_pillow_from_URL(i)) for i in raw_images]

    def __get_humans(self):
        cords = []
        for i in self.labels:
            c = []
            objects = i["object"]
            for obj in objects:
               if obj["title"] == "human":
                  c.append(obj["bbox"])
            cords.append(c) 
        return cords

    @staticmethod
    def get_pillow_from_URL(url):
        return Image.open(requests.get(url, stream=True).raw)


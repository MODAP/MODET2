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
        self.labels = [json.loads(i) for i in raw_labels]
        # self.images = [np.asarray(self.get_pillow_from_URL(i)) for i in raw_images]
        self.images = [(val[:, :, :-1] if (val := np.asarray(self.get_pillow_from_URL(i))).shape[2] != 3 else val) for i in raw_images]
        self.bbox = self.__get_humans(self.labels)

    @staticmethod
    def __get_humans(data):
        cords = []
        for i in data:
            c = []
            objects = i["objects"]
            for obj in objects:
               if obj["title"] == "human":
                   c.append(list(obj["bbox"].values()))
            cords.append(c) 
        return cords

    @staticmethod
    def get_pillow_from_URL(url):
        return Image.open(requests.get(url, stream=True).raw)


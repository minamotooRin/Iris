import pandas as pd
import httpx
import io

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path
from PIL import Image

good_title = {
    'brazil': 'Brazil',
    'turkey': 'Turkey',
    'united-states': 'USA',
    'india': 'India',
    'japan': 'Japan',
    'portugal': 'Portugal',
    'nigeria': 'Nigeria'
}
countries = ['brazil', 'turkey', 'united-states', 'india', 'japan', 'portugal', 'nigeria']


class Transcreations:
    def __init__(self, src, country_f, country_t, culture, offensive):

        self.data = {}

        self.data["path"] = src
        self.data["country_from"] = country_f
        self.data["country_to"] = country_t
        self.data["src_culture_score"] = culture
        self.data["src_offensive_score"] = offensive
        self.data["caption"] = None
        self.data["caption_edited"] = None
        self.data["transcreations"] = []

    def append(self, pa, name, vc, se, sp, na, cu, of):
        new_transcreation = {
            "path": pa,
            "method": name,
            "eval_human":
            {
                "visual_change": vc,
                "sem_eq": se,
                "spatial": sp,
                "natural": na,
                "culture": cu,
                "offensive": of
            },
            "eval_ai":
            {
                "visual_change": None,
                "sem_eq": None,
                "spatial": None,
                "natural": None,
                "culture": None,
                "offensive": None
            }
        }
        self.data["transcreations"].append(new_transcreation)

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        Info = ""
        for k, v in self.data.items():
            if k == "transcreations":
                continue
            Info += f"{k}: {v}\n"
        for ind, trans in enumerate(self.data["transcreations"]):
            Info += f"\t Transcreations {ind}:{trans['path']}\n"
        return Info


class TranscreationDataset():

    def __init__(self, path = "human_evaluation/"):

        human_eval_path = Path(path)

        self._dataset = { c: { c2: [] for c2 in countries if c != c2 } for c in countries }

        for c in countries:

            res = pd.read_csv(human_eval_path / f"{good_title[c]}" /"human_eval.csv")
            url = pd.read_csv(human_eval_path / f"{good_title[c]}" /"labels.csv")

            N = len(res)
            assert N == len(url)

            for i in range(N):
                row_res = res.iloc[i]
                row_url = url.iloc[i]

                visual_change2,visual_change3,visual_change4,sem_eq2,sem_eq3,sem_eq4,spatial2,spatial3,spatial4,natural2,natural3,natural4,culture1,culture2,culture3,culture4,offensive1,offensive2,offensive3,offensive4,split_number = row_res[:].values

                src_image_path,model_path_1,model_path_2,model_path_3,model_1,model_2,model_3 = row_url[:].values

                # src_image_path: https://storage.googleapis.com/image-transcreation/part1/india/beverages_kingfisher-beer
                src_country = src_image_path.split("/")[-2]

                obj = Transcreations(src_image_path, src_country, c, culture1, offensive1)
                obj.append(model_path_1, model_1, visual_change2, sem_eq2, spatial2, natural2, culture2, offensive2)
                obj.append(model_path_2, model_2, visual_change3, sem_eq3, spatial3, natural3, culture3, offensive3)
                obj.append(model_path_3, model_3, visual_change4, sem_eq4, spatial4, natural4, culture4, offensive4)

                self._dataset[src_country][c].append(obj)

    def __len__(self):
        return sum([len(self._dataset[c1][c2]) for c1 in self._dataset.keys() for c2 in self._dataset[c1].keys()])

    def __getitem__(self, key):
        return self._dataset[key]

    @property
    def dataset(self):
        return self._dataset

import json
import logging
import os
import shutil
from datetime import datetime
from typing import AnyStr, List, TypedDict

import numpy as np


class Config(TypedDict):
    train_folder: AnyStr
    validation_folder: AnyStr
    stride: int
    window_size: int
    batch_size: int
    validation: bool
    epochs: int
    augmentation: bool
    weights: AnyStr
    history: AnyStr
    lr: float
    mode: AnyStr

DEFAULT_CONFIG: Config = {
    "train_folder": "./train/",
    "validation_folder": "./validation/",
    "stride": 256,
    "window_size": 512,
    "batch_size": 1,
    "validation": False,
    "epochs": 1,
    "augmentation": True,
    "weights": None,
    "history": None,
    "mode": "RGB",
    "lr": 1e-4
}



def merge_json(config: Config, json) -> Config:
    if "train_folder" in json:
        config["train_folder"] = json["train_folder"]
    if "validation_folder" in json:
        config["validation_folder"] = json["validation_folder"]
    if "stride" in json:
        config["stride"] = json["stride"]
    if "window_size" in json:
        config["window_size"] = json["window_size"]
    if "batch_size" in json:
        config["batch_size"] = json["batch_size"]
    if "validation" in json:
        config["validation"] = json["validation"]
    if "epochs" in json:
        config["epochs"] = json["epochs"]
    if "augmentation" in json:
        config["augmentation"] = json["augmentation"]
    if "weights" in json:
        config["weights"] = json["weights"]
    if "history" in json:
        config["history"] = json["history"]
    if "lr" in json:
        config["lr"] = json["lr"]
    if "mode" in json:
        config["mode"] = json["mode"]
        
    return config


def load_config(config_filename) -> Config:
    config = DEFAULT_CONFIG
    try:
        if os.path.isfile(config_filename):
            with open(config_filename) as f:
                    json_prefs: Config = json.load(f)
                    config = merge_json(config, json_prefs)
    except:
        print("could not load preferences.json from {}".format(config_filename))

    return config


def save_config(config_filename, config):
    try:
        os.makedirs(os.path.dirname(config_filename), exist_ok=True)
        with open(config_filename, "w") as f:
            json.dump(config, f)
    except OSError as err:
        print("error serializing config")

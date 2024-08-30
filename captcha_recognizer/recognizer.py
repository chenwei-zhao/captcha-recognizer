import os
from pathlib import Path
from typing import Union

import numpy as np
from ultralytics import YOLO


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Recognizer(metaclass=SingletonMeta):
    def __init__(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(root_dir, 'captcha_recognizer', 'models', 'slider.onnx')
        self.model = YOLO(model_path, task='detect')

    def predict(self, source: Union[str, Path, int, list, tuple, np.ndarray] = None,
                **kwargs):
        results = self.model.predict(
            source=source,
            device='cpu',
            imgsz=[512, 512],
            classes=[0],
            **kwargs
        )
        if len(results):
            return results[0]

    def identify_gap(self, source, **kwargs):
        box = []
        conf = 0
        results = self.predict(source=source, **kwargs)
        if not len(results):
            return box, conf

        box_with_max_conf = max(results, key=lambda x: x.boxes.conf.max())

        box_with_conf = box_with_max_conf.boxes.data.tolist()[0]
        return box_with_conf[:-2], box_with_conf[-2]

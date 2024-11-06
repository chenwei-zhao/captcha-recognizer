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
            conf=0.8,
            imgsz=[416, 416],
            **kwargs
        )
        if len(results):
            return results[0]

    def identify_gap(self, source, show_result=False, **kwargs):
        box = []
        conf = 0
        results = self.predict(source=source, classes=[0], **kwargs)
        if not len(results):
            return box, conf

        box_with_max_conf = max(results, key=lambda x: x.boxes.conf.max())
        if show_result:
            box_with_max_conf.show()

        box_with_conf = box_with_max_conf.boxes.data.tolist()[0]
        return box_with_conf[:-2], box_with_conf[-2]

    def identify_screenshot(self, source, show_result=False, **kwargs):
        box = []
        conf = 0
        results = self.predict(source=source,
                               classes=[0, 2],
                               **kwargs)
        if not len(results):
            return box, conf

        slider = None
        slider_conf = 0

        box_list = []
        for result in results:

            if int(result.boxes.cls) == 2:
                current_slider_conf = float(result.boxes.conf[0])
                if current_slider_conf > slider_conf:
                    slider = result
            elif int(result.boxes.cls) == 0:
                box_list.append(result)

        if not slider or not box_list:
            return box, conf

        slider_row = slider.boxes.data.tolist()[0]
        slider_y_mid = int((slider_row[1] + slider_row[3]) / 2)

        box_with_nearest_y = min(box_list, key=lambda x: abs(
            (x.boxes.data.tolist()[0][1] + x.boxes.data.tolist()[0][3]) / 2 - slider_y_mid))

        if show_result:
            box_with_nearest_y.show()

        box_with_conf = box_with_nearest_y.boxes.data.tolist()[0]
        return box_with_conf[:-2], box_with_conf[-2]

    def identify_distance_by_screenshot(self, source, show_result=False, **kwargs):
        results = self.predict(source=source,
                               classes=[0, 2],
                               **kwargs)
        if not len(results):
            return

        slider = None
        slider_conf = 0

        box_list = []
        for result in results:

            if int(result.boxes.cls) == 2:
                current_slider_conf = float(result.boxes.conf[0])
                if current_slider_conf > slider_conf:
                    slider = result
            elif int(result.boxes.cls) == 0:
                box_list.append(result)

        if not slider or not box_list:
            return

        slider_row = slider.boxes.data.tolist()[0]
        slider_y_mid = int((slider_row[1] + slider_row[3]) / 2)
        slider_x = slider_row[0]

        box_with_nearest_y = min(box_list, key=lambda x: abs(
            (x.boxes.data.tolist()[0][1] + x.boxes.data.tolist()[0][3]) / 2 - slider_y_mid))

        if show_result:
            box_with_nearest_y.show()

        box_with_conf = box_with_nearest_y.boxes.data.tolist()[0]
        return int(box_with_conf[0] - slider_x)

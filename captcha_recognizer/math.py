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
        model_path = os.path.join(root_dir, 'captcha_recognizer', 'models', 'math.onnx')

        self.model = YOLO(model_path, task='detect')

    def extract_chars(self, results):
        chars = ''
        for result in results:

            data = list(result.boxes.data)

            # 按照 x1 值从小到大排序
            data.sort(key=lambda x: x[0])
            for row in data:
                class_id = int(row[5])
                name = self.model.names[class_id]
                chars += name

        return chars

    def identify_math(self, source, **kwargs):
        """
        识别给定图片的缺口。

        参数:
        - source: 图片源。
        - **kwargs: 其他传递给预测函数的参数。

        返回:
        - chars: 预测字符
        """

        default_kwargs = {
            'conf': 0.6,
            'imgsz': 224,
            'show_conf': False,
            'verbose': False

        }
        default_kwargs.update(kwargs)

        results = self.model.predict(source=source, **default_kwargs)
        if not results:
            return ''
        return self.extract_chars(results)

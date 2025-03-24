import os
from pathlib import Path
from typing import Union

import numpy as np
from ultralytics import YOLO

DEFAULT_CONF = 0.25


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Recognizer(metaclass=SingletonMeta):
    def __init__(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        multi_cls_model_path = os.path.join(root_dir, 'captcha_recognizer', 'models', 'multi_cls.onnx')
        single_cls_model_path = os.path.join(root_dir, 'captcha_recognizer', 'models', 'single_cls.onnx')

        self.multi_cls_model = YOLO(multi_cls_model_path, task='detect')
        self.single_cls_model = YOLO(single_cls_model_path, task='detect')

    @staticmethod
    def predict(model, source: Union[str, Path, int, list, tuple, np.ndarray] = None,
                **kwargs):

        params = {'source': source,
                  'device': 'cpu',
                  'conf': 0.8,
                  'imgsz': [416, 416]
                  }
        params.update(kwargs)
        results = model.predict(**params)
        if len(results):
            return results[0]
        return []

    def identify_gap(self, source, show_result=False, **kwargs):
        box = []
        box_conf = 0
        results = self.predict(model=self.multi_cls_model, source=source, classes=[0], conf=DEFAULT_CONF, **kwargs)
        if not len(results):
            return box, box_conf

        box_with_max_conf = max(results, key=lambda x: x.boxes.conf.max())
        if show_result:
            box_with_max_conf.show()

        box_with_conf = box_with_max_conf.boxes.data.tolist()[0]
        return box_with_conf[:-2], box_with_conf[-2]

    # 通过宽度和高度，相差的比例，按照权重1:1计算差异值
    @staticmethod
    def calculate_difference(slider, box):

        slider_with_conf = slider.boxes.data.tolist()[0]
        box_with_conf = box.boxes.data.tolist()[0]

        slider_height_mid = int((slider_with_conf[1] + slider_with_conf[3]) / 2)
        box_height_mid = int((box_with_conf[1] + box_with_conf[3]) / 2)

        width_slider = slider_with_conf[2] - slider_with_conf[0]
        height_slider = slider_with_conf[3] - slider_with_conf[1]

        width_box = box_with_conf[2] - box_with_conf[0]
        height_box = box_with_conf[3] - box_with_conf[1]

        # 通过中间点高度，物体宽度，物体高度计算差异值
        return abs(box_height_mid - slider_height_mid) * 2 + abs(width_box - width_slider) + abs(
            height_box - height_slider)

    def identify_boxes_by_screenshot(self, source, **kwargs):
        # 通过截图图片识别所有box
        results = self.predict(model=self.single_cls_model, source=source,
                               **kwargs)

        box_list = []
        if not len(results):
            return box_list

        for result in results:
            box_list.append(result)

        # 按x轴坐标排序，从小到大
        box_list.sort(key=lambda x: x.boxes.data.tolist()[0][0])
        return box_list

    def identify_target_boxes_by_screenshot(self, source, **kwargs):
        # 识别滑块框和目标缺口框

        slider_box = box_nearest = None

        box_list = self.identify_boxes_by_screenshot(source, **kwargs)

        if not box_list or len(box_list) == 1:
            return slider_box, box_nearest

        slider_box = box_list[0]

        others = box_list[1:]

        box_nearest = None
        min_box_diff = None

        for box in others:
            box_diff = self.calculate_difference(slider_box, box)
            if not min_box_diff:
                min_box_diff = box_diff
                box_nearest = box
                continue
            if box_diff < min_box_diff:
                min_box_diff = box_diff
                box_nearest = box

        return slider_box, box_nearest

    def identify_screenshot(self, source, show_result=False, **kwargs):
        # 通过截图识别滑块缺口
        slider_box, box_nearest = self.identify_target_boxes_by_screenshot(source, **kwargs)
        if not slider_box or not box_nearest:
            return [], 0
        if show_result:
            box_nearest.show()

        box_with_conf = box_nearest.boxes.data.tolist()[0]
        return box_with_conf[:-2], box_with_conf[-2]

    def identify_distance_by_screenshot(self, source, show_result=False, **kwargs):
        # 通过截图识别滑块缺口, 计算缺口与滑块的距离，计算滑块需要滑动距离
        slider_box, box_nearest = self.identify_target_boxes_by_screenshot(source, **kwargs)
        if not slider_box or not box_nearest:
            return
        if show_result:
            box_nearest.show()

        box_with_conf = box_nearest.boxes.data.tolist()[0]
        slider_with_conf = slider_box.boxes.data.tolist()[0]
        return int(box_with_conf[0] - slider_with_conf[0])

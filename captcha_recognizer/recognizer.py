import os
from pathlib import Path
from typing import Union

import cv2.dnn
import numpy as np

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
NMS_THRESHOLD = 0.5
NAMES = {0: 't', 1: 'f', 2: 's'}


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

        self.multi_cls_model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(multi_cls_model_path)

        self.single_cls_model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(single_cls_model_path)

    @staticmethod
    def image_to_array(source: Union[str, Path, bytes, np.ndarray] = None):
        if isinstance(source, (str, Path)):
            # 从文件路径读取
            return cv2.imread(str(source))
        elif isinstance(source, bytes):
            # 从字节流读取
            np_arr = np.frombuffer(source, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif isinstance(source, np.ndarray):
            # 如果已经是 numpy 数组，直接使用
            return source
        else:
            raise TypeError("Unsupported source type. Only str, Path, bytes, or numpy.ndarray are supported.")

    def predict(self, model, source: Union[str, Path, bytes, np.ndarray] = None, conf=CONF_THRESHOLD):

        # Read the input image
        original_image: np.ndarray = self.image_to_array(source)
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / 416

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(416, 416), swapRB=True)
        model.setInput(blob)

        # Perform inference
        outputs = model.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= conf:
                box = [
                    int((outputs[0][i][0] - (0.5 * outputs[0][i][2])) * scale),
                    int((outputs[0][i][1] - (0.5 * outputs[0][i][3])) * scale),
                    int((outputs[0][i][0] + (0.5 * outputs[0][i][2])) * scale),
                    int((outputs[0][i][1] + (0.5 * outputs[0][i][3])) * scale),
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD, NMS_THRESHOLD)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels

        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": NAMES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)

        return detections

    def identify_gap(self, source, is_single=False, conf=CONF_THRESHOLD, **kwargs):
        """
        识别给定图片的缺口。

        参数:
        - source: 图片源。
        - is_single: 布尔值，指示是否为单缺口图片。
        - conf: 置信度

        返回:
        - box: 一个列表，包含具有最高置信度的间隙的边界框坐标。
        - box_conf: 浮点数，代表间隙的置信度。
        """
        if is_single:
            model = self.single_cls_model
            classes = [0, 1, 2]
        else:
            model = self.multi_cls_model
            classes = [0]
        results = self.predict(model=model, source=source, conf=conf)
        box = []
        box_conf = 0
        if not len(results):
            return box, box_conf

        results_filtered = [result for result in results if result['class_id'] in classes]
        box_with_max_conf = max(results_filtered, key=lambda x: x['confidence'])

        return box_with_max_conf['box'], box_with_max_conf['confidence']

    # 通过宽度和高度，相差的比例，按照权重1:1计算差异值
    @staticmethod
    def calculate_difference(slider, box):

        slider_box = slider['box']
        box_loop = box['box']

        slider_height_mid = int((slider_box[1] + slider_box[3]) / 2)
        box_height_mid = int((box_loop[1] + box_loop[3]) / 2)

        width_slider = slider_box[2] - slider_box[0]
        height_slider = slider_box[3] - slider_box[1]

        width_box = box_loop[2] - box_loop[0]
        height_box = box_loop[3] - box_loop[1]

        # 通过中间点高度，物体宽度，物体高度计算差异值
        return abs(box_height_mid - slider_height_mid) * 2 + abs(width_box - width_slider) + abs(
            height_box - height_slider)

    def identify_boxes_by_screenshot(self, source: Union[str, Path, bytes, np.ndarray]):
        # 通过截图图片识别所有box
        results = self.predict(model=self.single_cls_model, source=source)

        box_list = []
        if not len(results):
            return box_list

        for result in results:
            box_list.append(result)

        # 按x轴坐标排序，从小到大
        box_list.sort(key=lambda x: x['box'][0])
        return box_list

    def identify_target_boxes_by_screenshot(self, source):
        # 识别滑块框和目标缺口框

        slider_box = box_nearest = None

        box_list = self.identify_boxes_by_screenshot(source)

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

    def identify_screenshot(self, source, **kwargs):
        # 通过截图识别滑块缺口
        slider_box, box_nearest = self.identify_target_boxes_by_screenshot(source)
        if not slider_box or not box_nearest:
            return [], 0

        return box_nearest['box'], box_nearest['confidence']

    def identify_distance_by_screenshot(self, source, **kwargs):
        # 通过截图识别滑块缺口, 计算缺口与滑块的距离，计算滑块需要滑动距离
        slider_box, box_nearest = self.identify_target_boxes_by_screenshot(source)
        if not slider_box or not box_nearest:
            return

        box_loop = box_nearest['box']
        slider_box = slider_box['box']
        return int(box_loop[0] - slider_box[0])

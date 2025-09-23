import os
import random
import time
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from shapely.geometry import Polygon

CONF_THRESHOLD = 0.25

IOU_THRESHOLD = 0.8

Y_IOU_THRESHOLD = 0.85


class Slider:

    def __init__(self):
        """
        Initialize the instance segmentation model using an ONNX model.
        """
        root_dir = os.path.dirname(os.path.dirname(__file__))
        slider_model_path = os.path.join(root_dir, 'captcha_recognizer', 'models', 'slider.onnx')

        self.session = ort.InferenceSession(
            slider_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == 'GPU' else [
                "CPUExecutionProvider"],
        )

        self.classes = {0: 's'}

    def predict(self, img: np.ndarray, conf: float = 0.25, iou: float = 0.7,
                imgsz: Union[int, Tuple[int, int]] = 640) -> List:
        """
        Run inference on the input image using the ONNX model.
        """
        imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        prep_img = self.preprocess(img, imgsz)
        outs = self.session.run(None, {self.session.get_inputs()[0].name: prep_img})
        return self.postprocess(img, prep_img, outs, conf=conf, iou=iou)

    @staticmethod
    def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Resize and pad image while maintaining aspect ratio.
        Returns exactly new_shape sized image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Calculate ratio and new dimensions
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        # Ensure new dimensions are at least 1 and not larger than target
        new_unpad = (max(1, min(new_unpad[0], new_shape[1])),
                     max(1, min(new_unpad[1], new_shape[0])))

        # Resize if needed
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Calculate padding
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = float(dw), float(dh)

        # Divide padding into 2 sides
        top, bottom = int(round(dh / 2)), int(round(dh / 2))
        left, right = int(round(dw / 2)), int(round(dw / 2))

        # Add padding
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Final check to ensure exact size (might need crop if rounding caused overflow)
        if img.shape[0] != new_shape[0] or img.shape[1] != new_shape[1]:
            img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

        return img

    def preprocess(self, img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """
        Preprocess the input image before feeding it into the model.
        """
        img = self.letterbox(img, new_shape)
        img = img[..., ::-1].transpose([2, 0, 1])[None]
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255
        return img

    def postprocess(self, img: np.ndarray, prep_img: np.ndarray, outs: List, conf: float = 0.25,
                    iou: float = 0.7) -> List:
        """
        Post-process model predictions to extract meaningful results.
        """
        preds, protos = outs
        preds = self.non_max_suppression(preds, conf, iou, nc=len(self.classes))

        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = self.scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)
            masks = self.process_mask(protos[i], pred[:, 6:], pred[:, :4], img.shape[:2])
            results.append([pred[:, :6], masks])

        return results

    def process_mask(self, protos: np.ndarray, masks_in: np.ndarray, bboxes: np.ndarray,
                     shape: Tuple[int, int]) -> np.ndarray:
        c, mh, mw = protos.shape
        masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
        masks = self.scale_masks(masks, shape)
        masks = self.crop_mask(masks, bboxes)
        return masks > 0.0

    @staticmethod
    def masks_to_segments(masks: Union[np.ndarray,], strategy: str = "largest") -> List[np.ndarray]:
        """
        将二值Mask转换为多边形边界点(segments)，不使用多边形简化

        参数:
            masks: 输入的二值Mask，可以是numpy数组或torch张量
                  形状为(batch_size, height, width)或(height, width)
            strategy: 处理多个轮廓的策略:
                     'all' - 合并所有轮廓
                     'largest' - 只保留最大轮廓
                     'none' - 返回所有轮廓不合并

        返回:
            包含多边形点集的列表，每个元素是(N,2)的numpy数组
        """
        # 转换输入为numpy数组

        masks_np = masks.astype("uint8")

        # 处理单张mask的情况
        if masks_np.ndim == 2:
            masks_np = masks_np[np.newaxis, ...]

        segments = []

        for mask in masks_np:
            # 查找轮廓 (OpenCV 4.x返回格式)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:  # 没有找到轮廓
                segments.append(np.zeros((0, 2), dtype=np.float32))
                continue

            # 根据策略处理多个轮廓
            if strategy == "all" and len(contours) > 1:
                # 合并所有轮廓，保留所有点
                contour = np.concatenate([x.reshape(-1, 2) for x in contours])
            elif strategy == "largest":
                # 选择最长的轮廓，保留所有点
                contour = max(contours, key=lambda x: cv2.arcLength(x, closed=True))
                contour = contour.reshape(-1, 2)
            else:  # 'none'策略或其他情况
                # 不合并轮廓，保留所有点
                contour = contours[0].reshape(-1, 2)

            segments.append(contour.astype(np.float32))

        return segments[0] if masks_np.shape[0] == 1 else segments

    @staticmethod
    def draw_segments(image, boxes, masks,
                      mask_alpha=0.5, box_thickness=2, draw_labels=True):

        """
        在图像上绘制预测框和掩膜

        参数:
            image: 原始图像 (numpy数组, BGR格式)
            boxes: 预测框列表, 格式为 [[x1, y1, x2, y2, score, class_id], ...]
            masks: 掩膜列表, 每个掩膜为二值图像 (0或255)
            box_color: 框的颜色 (BGR格式), 如果为None则随机生成
            mask_alpha: 掩膜透明度 (0-1)
            box_thickness: 框的线宽
            draw_labels: 是否绘制类别和置信度标签

        返回:
            绘制后的图像
        """
        # 创建输出图像的副本
        output = image.copy()

        # 如果没有提供boxes和masks，直接返回原图
        if boxes is None and masks is None:
            return output

        # 绘制masks
        if masks is not None:
            # 创建一个空的彩色掩膜图像
            color_mask = np.zeros_like(image)

            for i, mask in enumerate(masks):
                # 为每个mask生成随机颜色或使用指定颜色

                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                # 将二值mask转换为彩色mask
                mask = mask.astype(bool)
                color_mask[mask] = color

            # 将彩色掩膜与原始图像混合
            output = cv2.addWeighted(output, 1, color_mask, mask_alpha, 0)

        # 绘制boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2, score, class_id = box[:6]  # 只取前6个值，兼容不同格式

                # 为每个box生成随机颜色或使用指定颜色

                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # 绘制矩形框
                cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, box_thickness)

                # 绘制标签
                if draw_labels:
                    label = f"{int(class_id)}: {score:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    # 绘制标签背景
                    cv2.rectangle(output, (int(x1), int(y1) - label_height - 5),
                                  (int(x1) + label_width, int(y1)), color, -1)
                    # 绘制标签文本
                    cv2.putText(output, label, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return output

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

    @staticmethod
    def normalize_points(points):
        """
        将点集归一化到以原点为中心
        :param points: 点集
        :return: 归一化后的点集
        """
        # 计算质心
        centroid = np.mean(points, axis=0)
        # 将质心移到原点
        normalized_points = points - centroid
        return normalized_points

    @staticmethod
    def y_iou(segment1, segment2):
        # 计算交集
        start = max(segment1[0], segment2[0])
        end = min(segment1[1], segment2[1])
        intersection = max(0, end - start)  # 确保没有负值（无重叠时返回0）

        # 计算并集
        len1 = segment1[1] - segment1[0]
        len2 = segment2[1] - segment2[0]
        union = len1 + len2 - intersection

        # 计算 IoU
        iou = intersection / union if union != 0 else 0  # 避免除以0
        return iou

    def polygon_iou(self, poly1, poly2):
        """
        计算两个多边形的 IoU
        :param poly1: 多边形1的顶点坐标，格式为 [[x1,y1], [x2,y2], ..., [xn,yn]]
        :param poly2: 多边形2的顶点坐标，格式同上
        :return: IoU 值（范围 [0, 1]）
        """
        # 归一化处理到原点
        p1 = self.normalize_points(poly1)
        p2 = self.normalize_points(poly2)

        poly1 = Polygon(p1).buffer(0)  # buffer(0) 修复无效多边形（如自相交）
        poly2 = Polygon(p2).buffer(0)
        # poly2 = Polygon(normalize_points(poly2))

        # if not poly1.is_valid or not poly2.is_valid:
        #     return 0.0  # 无效多边形（如面积为零）

        # 计算交集和并集面积
        intersect = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        # 计算 IoU
        iou = intersect / union if union > 0 else 0.0
        return iou

    def pick_out_mask(self, boxes: list, segments):
        # boxes, masks 为两个列表，找出box值最小的一个
        box_slider = min(boxes, key=lambda x: x[0])
        box_slider_index = boxes.index(box_slider)
        segment_slider = segments[box_slider_index]

        box_sample = boxes[:box_slider_index] + boxes[box_slider_index + 1:]
        segment_sample = segments[:box_slider_index] + segments[box_slider_index + 1:]

        # 先按照y值iou过滤
        box_filtered = []
        segment_filtered = []

        for index, box in enumerate(box_sample):
            if self.y_iou([box_slider[1], box_slider[3]], [box[1], box[3]]) > Y_IOU_THRESHOLD:
                box_filtered.append(box)
                segment_filtered.append(segment_sample[index])
        # 如果通过y轴iou没有过滤掉有效值，则从所有box中选择iou最大的一个
        if not box_filtered:
            box_filtered = box_sample
            segment_filtered = segment_sample

        if len(box_filtered) == 1:
            return box_filtered[0], segment_filtered[0]

        iou_flag = 0
        iou_index = 0
        for index, segment in enumerate(segment_filtered):
            segment_iou = self.polygon_iou(segment_slider, segment)
            if segment_iou > iou_flag:
                iou_flag = segment_iou
                iou_index = index

        return box_filtered[iou_index], segment_filtered[iou_index]

    def identify(self, source: Union[str, Path, bytes, np.ndarray], conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, show=False):
        box_list = []
        mask_ndarray = None

        original_image: np.ndarray = self.image_to_array(source)
        results = self.predict(original_image, conf=conf, iou=iou, imgsz=640)

        if results:
            boxes, masks = results[0]
            if len(boxes) == 0:
                pass
            elif len(boxes) == 1:
                box_list = boxes[0].tolist()
                mask_ndarray = masks[0]

            else:
                segments = self.masks_to_segments(masks)
                box_list, _ = self.pick_out_mask(boxes.tolist(), segments)
                mask_ndarray = masks[boxes.tolist().index(box_list)]

        # 仅展示目标缺口
        if show and box_list and mask_ndarray is not None:
            sample = self.draw_segments(original_image, [box_list, ], [mask_ndarray, ])
            cv2.imshow('result', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if box_list:
            box = box_list[:4]
            box_conf = float(box_list[4])
        else:
            box = []
            box_conf = 0.0
        return box, box_conf

    def identify_offset(self, source: Union[str, Path, bytes, np.ndarray], conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                        show=False):
        """
        通过滑块图或者全图获取offset
        """
        box_list = []
        mask_ndarray = None

        original_image: np.ndarray = self.image_to_array(source)
        results = self.predict(original_image, conf=conf, iou=iou, imgsz=640)

        if results:
            boxes, masks = results[0]
            if len(boxes) == 0:
                pass
            elif len(boxes) == 1:
                box_list = boxes[0].tolist()
                mask_ndarray = masks[0]

            else:
                # 如果有多个目标，则选择X值最小的目标
                box_left = min(boxes, key=lambda x: x[0])
                box_list = box_left.tolist()
                mask_ndarray = masks[boxes.tolist().index(box_list)]

        # 仅展示目标缺口
        if show and box_list and mask_ndarray is not None:
            sample = self.draw_segments(original_image, [box_list, ], [mask_ndarray, ])
            cv2.imshow('result', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if box_list:
            box = box_list[:4]
            box_conf = float(box_list[4])
            offset = box[0]
        else:
            offset = 0
            box_conf = 0.0

        return offset, box_conf

    def scale_boxes(self, img1_shape: Tuple[int, int], boxes: np.ndarray, img0_shape: Tuple[int, int],
                    ratio_pad: Union[Tuple, None] = None, padding: bool = True, xywh: bool = False):
        """
        Rescale bounding boxes from one image shape to another.

        Rescales bounding boxes from img1_shape to img0_shape, accounting for padding and aspect ratio changes.
        Supports both xyxy and xywh box formats.

        Args:
            img1_shape (tuple): Shape of the source image (height, width).
            boxes (np.ndarray): Bounding boxes to rescale in format (N, 4).
            img0_shape (tuple): Shape of the target image (height, width).
            ratio_pad (tuple, optional): Tuple of (ratio, pad) for scaling. If None, calculated from image shapes.
            padding (bool): Whether boxes are based on YOLO-style augmented images with padding.
            xywh (bool): Whether box format is xywh (True) or xyxy (False).

        Returns:
            (np.ndarray): Rescaled bounding boxes in the same format as input.
        """
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (
                round((img1_shape[1] - img0_shape[1] * gain) / 2),
                round((img1_shape[0] - img0_shape[0] * gain) / 2),
            )
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        if padding:
            boxes[..., 0] -= pad[0]
            boxes[..., 1] -= pad[1]
            if not xywh:
                boxes[..., 2] -= pad[0]
                boxes[..., 3] -= pad[1]
        boxes[..., :4] /= gain
        return self.clip_boxes(boxes, img0_shape)

    @staticmethod
    def get_covariance_matrix(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate covariance matrix from oriented bounding boxes.

        Args:
            boxes (np.ndarray): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

        Returns:
            (np.ndarray): Covariance matrices corresponding to original rotated bounding boxes.
        """
        gbbs = np.concatenate((np.power(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), axis=-1)
        a, b, c = np.split(gbbs, [1, 2], axis=-1)
        cos = np.cos(c)
        sin = np.sin(c)
        cos2 = np.power(cos, 2)
        sin2 = np.power(sin, 2)
        return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

    def batch_probiou(self, obb1: np.ndarray, obb2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """
        Calculate the probabilistic IoU between oriented bounding boxes.

        Args:
            obb1 (np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
            obb2 (np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
            eps (float, optional): A small value to avoid division by zero.

        Returns:
            (np.ndarray): A tensor of shape (N, M) representing obb similarities.
        """
        x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
        x2, y2 = (np.expand_dims(x.squeeze(-1), 0) for x in np.split(obb2[..., :2], 2, axis=-1))
        a1, b1, c1 = self.get_covariance_matrix(obb1)
        a2, b2, c2 = (np.expand_dims(x.squeeze(-1), 0) for x in self.get_covariance_matrix(obb2))

        t1 = (
                     ((a1 + a2) * np.power(y1 - y2, 2) + (b1 + b2) * np.power(x1 - x2, 2)) / (
                     (a1 + a2) * (b1 + b2) - np.power(c1 + c2, 2) + eps)
             ) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - np.power(c1 + c2, 2) + eps)) * 0.5

        term1_log = (a1 * b1 - np.power(c1, 2)).clip(0)
        term2_log = (a2 * b2 - np.power(c2, 2)).clip(0)

        denominator = 4 * np.sqrt(term1_log * term2_log) + eps
        t3_numerator = (a1 + a2) * (b1 + b2) - np.power(c1 + c2, 2)
        # 确保 log 的输入为正值
        t3_arg = np.clip(t3_numerator / denominator + eps, eps, None)
        t3 = np.log(t3_arg) * 0.5

        bd = (t1 + t2 + t3).clip(eps, 100.0)
        hd = np.sqrt(1.0 - np.exp(-bd) + eps)
        return 1 - hd

    def nms_rotated(self, boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.45):
        """
        Perform NMS on oriented bounding boxes using probiou and fast-nms.

        Args:
            boxes (np.ndarray): Rotated bounding boxes with shape (N, 5) in xywhr format.
            scores (np.ndarray): Confidence scores with shape (N,).
            threshold (float): IoU threshold for NMS.

        Returns:
            (np.ndarray): Indices of boxes to keep after NMS.
        """
        sorted_idx = np.argsort(scores)[::-1]
        boxes = boxes[sorted_idx]
        ious = self.batch_probiou(boxes, boxes)

        # 使用更高效的方式创建上三角矩阵
        n = boxes.shape[0]
        ious[np.tril_indices(n)] = 0  # 将下三角和对角线置零

        pick = np.where((ious >= threshold).sum(axis=0) <= 0)[0]
        return sorted_idx[pick]

    def clip_boxes(self, boxes: np.ndarray, shape: Tuple[int, int]):
        """
        Clip bounding boxes to image boundaries.

        Args:
            boxes (np.ndarray): Bounding boxes to clip.
            shape (tuple): Image shape as (height, width).

        Returns:
            (np.ndarray): Clipped bounding boxes.
        """
        boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, shape[1])
        boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, shape[0])
        return boxes

    @staticmethod
    def xywh2xyxy(x: np.ndarray):
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format.

        Args:
            x (np.ndarray): Input bounding box coordinates in (x, y, width, height) format.

        Returns:
            (np.ndarray): Bounding box coordinates in (x1, y1, x2, y2) format.
        """
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = np.empty_like(x, dtype=np.float32)
        xy = x[..., :2]
        wh = x[..., 2:] / 2
        y[..., :2] = xy - wh
        y[..., 2:] = xy + wh
        return y

    @staticmethod
    def crop_mask(masks: np.ndarray, boxes: np.ndarray):
        """
        Crop masks to bounding box regions.

        Args:
            masks (np.ndarray): Masks with shape (N, H, W).
            boxes (np.ndarray): Bounding box coordinates with shape (N, 4) in relative point form.

        Returns:
            (np.ndarray): Cropped masks.
        """
        _, h, w = masks.shape
        # 确保 boxes 的维度正确
        boxes = boxes[:, :, None] if boxes.ndim == 2 else boxes
        x1, y1, x2, y2 = np.split(boxes, 4, axis=1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask_np(self, protos: np.ndarray, masks_in: np.ndarray, bboxes: np.ndarray, shape: Tuple[int, int],
                        upsample: bool = False):
        """
        Apply masks to bounding boxes using mask head output.

        Args:
            protos (np.ndarray): Mask prototypes with shape (mask_dim, mask_h, mask_w).
            masks_in (np.ndarray): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
            bboxes (np.ndarray): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
            shape (tuple): Input image size as (height, width).
            upsample (bool): Whether to upsample masks to original image size.

        Returns:
            (np.ndarray): A binary mask array of shape [n, h, w], where n is the number of masks after NMS, and h and w
                are the height and width of the input image. The mask is applied to the bounding boxes.
        """
        c, mh, mw = protos.shape
        ih, iw = shape

        masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
        width_ratio = mw / iw
        height_ratio = mh / ih

        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= width_ratio
        downsampled_bboxes[:, 2] *= width_ratio
        downsampled_bboxes[:, 3] *= height_ratio
        downsampled_bboxes[:, 1] *= height_ratio

        masks = self.crop_mask(masks, downsampled_bboxes)
        if upsample:
            masks = cv2.resize(masks.transpose((1, 2, 0)),
                               (shape[1], shape[0]),
                               interpolation=cv2.INTER_LINEAR).transpose((2, 0, 1))

        return masks > 0.0

    @staticmethod
    def scale_masks(masks: np.ndarray, shape: Tuple[int, int], padding: bool = True):
        """
        Rescale segment masks to target shape.
        Args:
            masks (np.ndarray): Masks with shape (N, H, W).
            shape (tuple): Target height and width as (height, width).
            padding (bool): Whether masks are based on YOLO-style augmented images with padding.
        Returns:
            (np.ndarray): Rescaled masks with shape (N, H_new, W_new).
        """
        mh, mw = masks.shape[1:]
        gain = min(mh / shape[0], mw / shape[1])
        pad = [mw - shape[1] * gain, mh - shape[0] * gain]

        if padding:
            pad[0] /= 2
            pad[1] /= 2

        top, left = (int(round(pad[1])), int(round(pad[0]))) if padding else (0, 0)
        bottom, right = (
            mh - int(round(pad[1])),
            mw - int(round(pad[0])),
        )

        # Crop the masks first
        masks_cropped = masks[:, top:bottom, left:right]

        # 向量化 resize 操作
        resized_masks = np.zeros((masks_cropped.shape[0], shape[0], shape[1]), dtype=masks_cropped.dtype)
        for i, mask in enumerate(masks_cropped):
            resized_masks[i] = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

        return resized_masks

    def non_max_suppression(
            self,
            prediction: np.ndarray,
            conf_thres: float = 0.25,
            iou_thres: float = 0.45,
            classes=None,
            agnostic: bool = False,
            multi_label: bool = False,
            labels=(),
            max_det: int = 300,
            nc: int = 0,
            max_time_img: float = 0.05,
            max_nms: int = 30000,
            max_wh: int = 7680,
            in_place: bool = True,
            rotated: bool = False,
            end2end: bool = False,
            return_idxs: bool = False,
    ):
        """
        Perform non-maximum suppression (NMS) on prediction results.
        """
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        if classes is not None:
            classes = np.array(classes)

        if prediction.shape[-1] == 6 or end2end:
            output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
            if classes is not None:
                output = [pred[np.any(pred[:, 5:6] == classes, axis=1)] for pred in output]
            return output

        bs = prediction.shape[0]
        nc = nc or (prediction.shape[1] - 4)
        extra = prediction.shape[1] - nc - 4
        mi = 4 + nc
        xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres
        xinds = np.stack([np.arange(len(i)) for i in xc])[..., None]

        time_limit = 2.0 + max_time_img * bs
        multi_label &= nc > 1

        prediction = np.transpose(prediction, (0, 2, 1))
        if not rotated:
            if in_place:
                prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])
            else:
                prediction = np.concatenate((self.xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), axis=-1)

        t = time.time()
        output = [np.zeros((0, 6 + extra), dtype=np.float32)] * bs
        keepi = [np.zeros((0, 1), dtype=np.int64)] * bs
        for xi, (x, xk) in enumerate(zip(prediction, xinds)):
            filt = xc[xi]
            x, xk = x[filt], xk[filt]

            # 增强 labels 的健壮性
            if labels and len(labels) > xi and len(labels[xi]) and not rotated:
                lb = np.array(labels[xi])
                if lb.size > 0:
                    v = np.zeros((len(lb), nc + extra + 4), dtype=np.float32)
                    v[:, :4] = self.xywh2xyxy(lb[:, 1:5])
                    v[range(len(lb)), lb[:, 0].astype(np.int64) + 4] = 1.0
                    x = np.concatenate((x, v), axis=0)

            if not x.shape[0]:
                continue

            box, cls, mask = np.split(x, [4, 4 + nc], axis=1)

            if multi_label:
                i, j = np.where(cls > conf_thres)
                x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].astype(np.float32), mask[i]), axis=1)
                xk = xk[i]
            else:
                conf = np.amax(cls, axis=1, keepdims=True)
                j = np.argmax(cls, axis=1, keepdims=True)
                filt = conf.squeeze(-1) > conf_thres
                x = np.concatenate((box, conf, j.astype(np.float32), mask), axis=1)[filt]
                xk = xk[filt]

            if classes is not None:
                filt = np.any(x[:, 5:6] == classes, axis=1)
                x, xk = x[filt], xk[filt]

            n = x.shape[0]
            if not n:
                continue
            if n > max_nms:
                filt = np.argsort(x[:, 4])[::-1][:max_nms]
                x, xk = x[filt], xk[filt]

            c = x[:, 5:6] * (0 if agnostic else max_wh)
            scores = x[:, 4]

            if rotated:
                boxes = np.concatenate((x[:, :2] + c, x[:, 2:4], x[:, -1:]), axis=-1)
                i = self.nms_rotated(boxes, scores, iou_thres)
            else:
                boxes = x[:, :4] + c
                # Custom NMS for numpy
                i = []
                if boxes.shape[0] > 0:
                    y1, x1, y2, x2 = boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]
                    area = (x2 - x1) * (y2 - y1)
                    order = scores.argsort()[::-1]
                    while order.size > 0:
                        idx = order[0]
                        i.append(idx)
                        xx1 = np.maximum(x1[idx], x1[order[1:]])
                        yy1 = np.maximum(y1[idx], y1[order[1:]])
                        xx2 = np.minimum(x2[idx], x2[order[1:]])
                        yy2 = np.minimum(y2[idx], y2[order[1:]])
                        w = np.maximum(0.0, xx2 - xx1)
                        h = np.maximum(0.0, yy2 - yy1)
                        inter = w * h
                        iou = inter / (area[idx] + area[order[1:]] - inter)
                        order = order[np.where(iou <= iou_thres)[0] + 1]
                i = np.array(i)

            i = i[:max_det]

            output[xi], keepi[xi] = x[i], xk[i].reshape(-1)
            if (time.time() - t) > time_limit:
                break

        return (output, keepi) if return_idxs else output


if __name__ == "__main__":
    """
    单缺口
    """
    model = Slider()
    res = model.identify(source='img_example.png', show=True)
    print('results', res)

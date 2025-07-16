from captcha_recognizer.recognizer import Recognizer

# 1. 单缺口、多缺口验证码背景图识别
# source传入图片， 支持 Union[str, Path, bytes, np.ndarray]

recognizer = Recognizer()
box, confidence = recognizer.identify_gap(source='images_example/example1.png')

print(f'缺口坐标: {box}')
print(f'可信度: {confidence}')

"""
打印结果如下:
缺口方框坐标: [331.72052001953125, 55.96122741699219, 422.079345703125, 161.7498779296875]
可信度: 0.9513089656829834

坐标原点：图片左上角
缺口方框坐标为缺口方框左上角和右下角距离坐标原点的距离
"""


# 2. 验证码截图识别
# source传入图片， 支持 Union[str, Path, bytes, np.ndarray]

recognizer = Recognizer()
box, confidence = recognizer.identify_screenshot(source='images_example/example8.png')

print(f'缺口坐标: {box}')
print(f'可信度: {confidence}')
"""
打印结果如下:
缺口坐标: [332.2833251953125, 55.69723129272461, 422.9914245605469, 162.34860229492188]
可信度: 0.9590587019920349

# """

# 3. 基于验证码截图识别滑块距离

# source传入图片， 支持 Union[str, Path, bytes, np.ndarray]

recognizer = Recognizer()
distance = recognizer.identify_distance_by_screenshot(source='images_example/example8.png')

print('滑块距离', distance)

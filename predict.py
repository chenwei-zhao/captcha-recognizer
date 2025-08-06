# V1版本
from captcha_recognizer.recognizer import Recognizer

recognizer = Recognizer()

images_dir = 'images_example'
for i in range(1, 8):
    box, confidence = recognizer.identify_gap(source=f'images_example/example{i}.png')
    print(f'缺口坐标: {box}')
    print('置信度', confidence)

# # V2增强版
from captcha_recognizer.slider import SliderV2

images_dir = 'images_example'
for i in range(8, 9):
    box, confidence = SliderV2().identify(source=f'images_example/example{i}.png', show=True)
    print(f'缺口坐标: {box}')
    print('置信度', confidence)


from captcha_recognizer.slider import Slider

images_dir = 'images_example'
for i in range(4, 12):
    box, confidence = Slider().identify(source=f'images_example/example{i}.png', show=True)
    print(f'缺口坐标: {box}')
    print('置信度', confidence)

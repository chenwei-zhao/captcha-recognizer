from captcha_recognizer.recognizer import Recognizer

recognizer = Recognizer()

box, confidence = recognizer.identify_gap(source=f'images_example/example1.png')

print(f'缺口坐标: {box}')
print(f'可信度: {confidence}')

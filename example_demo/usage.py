from captcha_recognizer.recognizer import Recognizer

box, confidence = Recognizer().identify_gap(source='example_demo/example1.png', verbose=False)

print(f'缺口坐标: {box}')
print(f'可信度: {confidence}')

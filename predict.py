from captcha_recognizer.recognizer import Recognizer

recognizer = Recognizer()

images_dir = 'images_example'
for i in range(1, 8):
    box, confidence = recognizer.identify_gap(source=f'images_example/example{i}.png', save=True)
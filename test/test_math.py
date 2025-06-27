from captcha_recognizer.math import Recognizer

recognizer = Recognizer()

test_image = 'images_test/math.png'
chars = recognizer.identify_math(test_image)
print('chars', chars)

[简体中文](README.md) | English

# Captcha-Recognizer

Captcha-Recognizer is an easy-to-use universal slider recognition library that trains a universal breach detection model through deep learning. Based on the training results, it recognises the location of the slider breach in the verification code and returns the coordinates and confidence of the breach.

# Statement
This project is not for any verification code manufacturer, all the content of the project is only for learning and exchange, not for any other purpose, and it is strictly prohibited to use for illegal purposes.

# License

MIT license


# Version Requirements

* ``Python`` >=  3.8.0
* ``ultralytics`` >=  8.0.0
* ``torch`` >=  1.8.0
* ``onnxruntime``
* ``onnx``

* Works on Linux, Windows, macOS


# Install


From pip 

```bash
pip install captcha-recognizer
```



# Usage


```Python

from captcha_recognizer.recognizer import Recognizer

# source is your image path
box, confidence = Recognizer().identify_gap(source='example_demo/example1.png', verbose=False)

print(f'Gap coordinate: {box}')
print(f'Confidence: {confidence}')

"""
Gap coordinate: [331.72052001953125, 55.96122741699219, 422.079345703125, 161.7498779296875]
Confidence: 0.9513089656829834

Origin of coordinates: top left corner of the picture
The notch box coordinate is the distance between the top left corner and the bottom right corner of the notch box and the origin of the coordinate
"""
```

# Sample slider picture



Includes, but is not limited to, the following types

<p>example 1</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/example_demo/example1.png" alt="captcha" width="320">
<p>example 2</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/example_demo/example2.png" alt="captcha" width="320">
<p>example 3</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/example_demo/example3.png" alt="captcha" width="320">
<p>example 4</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/example_demo/example4.png" alt="captcha" width="320">




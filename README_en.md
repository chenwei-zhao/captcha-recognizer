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


# Usage

- [HTTP API](https://github.com/chenwei-zhao/captcha-api)
- From Pypi

## HTTP API
[HTTP API](https://github.com/chenwei-zhao/captcha-api)

## From Pypi
```bash
pip install captcha-recognizer
```


## Background image recognition
```Python

from captcha_recognizer.recognizer import Recognizer

# source is your image path, set verbose to be False to disable verbose log
recognizer = Recognizer()
box, confidence = recognizer.identify_gap(source='your_example_image.png', verbose=False)

print(f'Gap coordinate: {box}')
print(f'Confidence: {confidence}')

"""
Gap coordinate: [331.72052001953125, 55.96122741699219, 422.079345703125, 161.7498779296875]
Confidence: 0.9513089656829834

Origin of coordinates: top left corner of the picture
The notch box coordinate is the distance between the top left corner and the bottom right corner of the notch box and the origin of the coordinate
"""
```

## Screenshot recognition

```python
from captcha_recognizer.recognizer import Recognizer
recognizer = Recognizer()

box, confidence = recognizer.identify_screenshot(source='<image obj>', verbose=False, show_result=True)

print(f'Gap coordinate: {box}')
print(f'Confidence: {confidence}')

```


# Sample slider picture



Includes, but is not limited to, the following types


<p>example 1</p>
<p>尺寸 552*344</p>
<img 
  src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_example/example1.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example1.png"
>
<p>predict 1</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_predict/predict1.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict1.png"
>

<p>example 2</p>
<p>尺寸 260*160</p>p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_example/example2.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example2.png"
>

<p>predict 2</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_predict/predict2.png"
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict2.png"
>

<p>example 3</p>
<p>尺寸 400*200</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_example/example3.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example1.png"
>
<p>predict3</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_predict/predict3.png"
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict3.png"
>


<p>example 4</p>
<p>尺寸 672*390</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_example/example4.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example4.png"
>
<p>predict4</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_predict/predict4.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict4.png"
>

<p>example 5</p>
<p>尺寸 280*155</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_example/example5.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example5.png"
>

<p>predict 5</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_predict/predict5.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict5.png"
>

<p>example 6</p>
<p>尺寸 590*360</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_example/example6.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example6.png"
>
<p>predict 6</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_predict/predict6.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict6.png"
>

<p>example 7</p>
<p>尺寸 320*160</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_example/example7.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example7.png"
>
<p>predict 7</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_predict/predict7.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict7.png"
>



# have a problem
- Error loading “xxx\Lib\site-packages\torch\lib\fbgemm.dll” or one of its dependencies.
  - See [Issues 2](https://github.com/chenwei-zhao/captcha-recognizer/issues/2)
- Model Unsupported model IR version: 9, max supported IR version: 8
    - See [Issues 1](https://github.com/chenwei-zhao/captcha-recognizer/issues/1)

# Email
- Gmail: chenwei.zhaozhao@gmail.com
- 163: chenwei_nature@163.com
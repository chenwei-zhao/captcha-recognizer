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
<img 
  src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/example_demo/example1.png" 
  alt="captcha" 
  width="320"
  onerror="this.src = 'https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example1.png'"
>
<p>效果图1</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/example_demo/predict1.png" 
  alt="captcha" 
  width="320"
  onerror="this.src = 'https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict1.png'"
>

<p>example 2</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/example_demo/example2.png" 
  alt="captcha"
  width="320"
  onerror="this.src = 'https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example2.png'"
>

<p>效果图2</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/example_demo/predict2.png"
  alt="captcha"
  width="320"
  onerror="this.src = 'https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict2.png'"
>

<p>example 3</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/example_demo/example3.png" 
  alt="captcha"
  width="320"
  onerror="this.src = 'https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example3.png'"
>
<p>效果图3</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/example_demo/predict3.png" 
  alt="captcha"
  width="320"
  onerror="this.src = 'https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict3.png'"
>


<p>example 4</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/example_demo/example4.png" 
  alt="captcha"
  width="320"
  onerror="this.src = 'https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example4.png'"
>
<p>效果图4</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/example_demo/predict4.png" 
  alt="captcha"
  width="320"
  onerror="this.src = 'https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict4.png'"
>



# have a problem
- Error loading “xxx\Lib\site-packages\torch\lib\fbgemm.dll” or one of its dependencies.
  - See [Issues 2](https://github.com/chenwei-zhao/captcha-recognizer/issues/2)
- Model Unsupported model IR version: 9, max supported IR version: 8
    - See [Issues 1](https://github.com/chenwei-zhao/captcha-recognizer/issues/1)

# Email
- Gmail: chenwei.zhaozhao@gmail.com
- 163: chenwei_nature@163.com
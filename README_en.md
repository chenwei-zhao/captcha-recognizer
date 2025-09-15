[简体中文](README.md) | English

# Captcha-Recognizer

Captcha-Recognizer is an easy-to-use universal slider recognition library that trains a universal breach detection model through deep learning. Based on the training results, it recognises the location of the slider breach in the verification code and returns the coordinates and confidence of the breach.

# Statement
This project is not for any verification code manufacturer, all the content of the project is only for learning and exchange, not for any other purpose, and it is strictly prohibited to use for illegal purposes.

# License

MIT license


# Version Requirements

* ``Python`` >=  3.6.0
* ``opencv-python``
* ``shapely``
* ``onnxruntime``

* Works on Linux, Windows, MacOS


# Usage

- [HTTP API](https://github.com/chenwei-zhao/captcha-api)
- From Pypi

## HTTP API
[HTTP API](https://github.com/chenwei-zhao/captcha-api)

## From Pypi
```bash
pip install captcha-recognizer
```


## Example Code

```python
from captcha_recognizer.slider import Slider

box, confidence = Slider().identify(source=f'images_example/example8.png', show=True)
print(f'Gap coordinate: {box}')
print('Confidence', confidence)

```

# Sample slider picture


Includes, but is not limited to, the following types


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

<p>example 8</p>
<img 
  src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_example/example8.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example8.png"
>
<p>predict 8</p>
<img src="https://raw.githubusercontent.com/chenwei-zhao/captcha-recognizer/main/images_predict/predict8.png" 
  alt="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict8.png"
>



# Version about opencv-python and numpy

Version 1：

opencv-python==4.12.0.88
numpy==2.2.6

Version 2：
opencv-python==4.6.0.66
numpy==1.24.4

# Email
- Gmail: chenwei.zhaozhao@gmail.com
- 163: chenwei_nature@163.com
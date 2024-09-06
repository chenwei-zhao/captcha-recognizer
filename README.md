简体中文 | [English](README_en.md)

# Captcha-Recognizer

Captcha-Recognizer是一个易用的通用滑块验证码识别库，通过深度学习训练通用的缺口检测模型，基于训练的结果，识别出验证码中的滑块缺口位置，并返回缺口的坐标与可信度。

# 声明
本项目不针对任何一家验证码厂商，项目所有内容仅供学习交流使用，不用于其他任何目的，严禁用于非法用途。


# 许可证

MIT license


# 版本要求

* ``Python`` >=  3.8.0
* ``ultralytics`` >=  8.0.0
* ``torch`` >=  1.8.0
* ``onnxruntime``
* ``onnx``

* Works on Linux, Windows, macOS


# 安装


From pip 

```bash
pip install captcha-recognizer
```



# 使用示例


```Python

from captcha_recognizer.recognizer import Recognizer

# source传入图片路径
recognizer = Recognizer()
box, confidence = recognizer.identify_gap(source='your_example_image.png',)

print(f'缺口坐标: {box}')
print(f'可信度: {confidence}')

"""
打印结果如下:
缺口方框坐标: [331.72052001953125, 55.96122741699219, 422.079345703125, 161.7498779296875]
可信度: 0.9513089656829834

坐标原点：图片左上角
缺口方框坐标为缺口方框左上角和右下角距离坐标原点的距离
"""
```

# 示例图片

包括且不限于以下类型的滑块图片检测

<p>示例图1</p>
<img 
  src="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example1.png" 
  alt="captcha" 
  width="320"
>
<p>效果图1</p>
<img src="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict1.png" 
  alt="captcha" 
  width="320"
>

<p>示例图2</p>
<img src="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example2.png" 
  alt="captcha"
  width="320"
>

<p>效果图2</p>
<img src="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict2.png"
  alt="captcha"
  width="320"
>

<p>示例图3</p>
<img src="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example3.png" 
  alt="captcha"
  width="320"
>
<p>效果图3</p>
<img src="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict3.png" 
  alt="captcha"
  width="320"
>


<p>示例图4</p>
<img src="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/example4.png" 
  alt="captcha"
  width="320"
>
<p>效果图4</p>
<img src="https://captcha-slider.oss-cn-beijing.aliyuncs.com/slider/predict4.png" 
  alt="captcha"
  width="320"
>


# 遇到问题
- Error loading “xxx\Lib\site-packages\torch\lib\fbgemm.dll” or one of its dependencies.
  - 参考 [Issues 2](https://github.com/chenwei-zhao/captcha-recognizer/issues/2)
- Model Unsupported model IR version: 9, max supported IR version: 8
    - 参考 [Issues 1](https://github.com/chenwei-zhao/captcha-recognizer/issues/1)
    


# 项目维护

- 本项目长期维护。
- 如果您有任何问题，欢迎提[issue](https://github.com/chenwei-zhao/captcha-recognizer/issues)。
- 如果您遇到本项目不能识别的滑块验证码，欢迎提issue，我会尽快解决。

# 更多联系方式
- Gmail: chenwei.zhaozhao@gmail.com
- 163/网易: chenwei_nature@163.com

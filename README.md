简体中文 | [English](README_en.md)

# Captcha-Recognizer

Captcha-Recognizer是一个易用的通用滑块验证码识别库，通过深度学习训练通用的缺口检测模型，基于训练的结果，识别出验证码中的滑块缺口位置，并返回缺口的坐标与可信度。

# 声明
本项目不针对任何一家验证码厂商，项目所有内容仅供学习交流使用，不用于其他任何目的，严禁用于非法用途。


# 许可证

MIT license


# 版本要求

* `Python`` >=  3.8.0
* ``ultralytics`` >=  8.0.0
* ``torch`` >=  1.8.0

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
box, confidence = Recognizer().identify_gap(source='example_demo/example1.png', verbose=False)

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

**参考例图**

包括且不限于以下类型的滑块图片检测

![Example Image 1](/example_demo/example1.png)
![Example Image 2](/example_demo/example2.png)
![Example Image 3](/example_demo/example3.png)
![Example Image 4](/example_demo/example4.png)
 




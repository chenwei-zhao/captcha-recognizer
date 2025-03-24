from setuptools import find_packages, setup


def read_file(filename):
    with open(filename, encoding='utf-8') as fp:
        return fp.read().strip()


def read_requirements(filename):
    return [
        line.strip()
        for line in read_file(filename).splitlines()
        if not line.startswith("#")
    ]


setup(
    name='captcha-recognizer',
    version='0.7.1',
    description='滑块验证码识别，基于YOLOv8训练，支持单缺口、多缺口、截图识别',
    long_description=read_file("README.md") + "\n\n" + read_file("HISTORY.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/chenwei-zhao/captcha-recognizer",
    author='Zhao Chenwei',
    author_email='chenwei.zhaozhao@gmail.com',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    keywords=["captcha", "slider", "captcha-recognizer", "captcha_recognizer", "滑块", "滑块验证码", "滑块识别"],
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=read_requirements("requirements.txt")
)

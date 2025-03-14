from setuptools import setup, find_packages

setup(
    name="msformer",
    version="0.1.0",
    packages=find_packages(),
    author="Bingjie Zhu e-mail:zhubj@zju.edu.cn",
    description="a novel molecular representation framework via meta structures",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ZJUFanLab/msformer",
)
from setuptools import setup,find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

#with open("requirement.txt") as fr:
#    requirement = fr.readlines()

setup(
    name="exkaldi",
    version="0.1",
    author="Yu Wang",
    author_email="wangyu@alps-lab.org",
    description="ExKaldi Automatic Speech Recognition Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wangyu09/exkaldi",
    packages=find_packages(),
    install_requires=["numpy>=1.16","PyAudio>=0.2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License:: 2.0",
        "Operating System :: OS Independent",
    ],
)
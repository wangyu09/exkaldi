from setuptools import setup,find_packages
import glob

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

#with open("requirement.txt") as fr:
#    requirement = fr.readlines()

setup(
    name="exkaldi",
    version="1.0.1",
    author="Wang Yu",
    author_email="wangyu@alps-lab.org",
    description="ExKaldi Automatic Speech Recognition Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wangyu09/exkaldi",
    packages=find_packages(),
    data_files = [
            ("exkaldisrc/tools", glob.glob("tools/*"))
        ],
    install_requires=["numpy>=1.16","PyAudio>=0.2", "kenlm>=0.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)

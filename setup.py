from setuptools import setup,find_packages
import glob
import os
import subprocess

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

#with open("requirement.txt") as fr:
#    requirement = fr.readlines()

def read_version_info():
    cmd = 'cd exkaldi && python -c "import version; print(version.version.version)"'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print(err.decode())
        raise Exception("Detect version error.")
    else:
        return out.decode().strip().split("\n")[-1].strip()

setup(
    name="exkaldi",
    version=read_version_info(),
    author="Wang Yu",
    author_email="wangyu@alps-lab.org",
    description="ExKaldi Automatic Speech Recognition Toolkit",
    long_description=long_description,
    long_description_content_type=os.path.join("text","markdown"),
    url="https://github.com/wangyu09/exkaldi",
    packages=find_packages(),
    data_files = [
            (os.path.join("exkaldisrc","tools"), glob.glob( os.path.join("tools","*")))
        ],
    install_requires=["numpy>=1.16", "PyAudio>=0.2", "kenlm>=0.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)

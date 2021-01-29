from setuptools import setup,find_packages
import glob
import os
import sys
import subprocess

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

#with open("requirement.txt") as fr:
#    requirement = fr.readlines()

def read_version_info():
    cmd = 'cd exkaldi && python3 -c "import version; print(version.info.version)"'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise Exception("Failed to detect ExKaldi version.\n"+err.decode())
    else:
        return out.decode().strip().split("\n")[-1].strip()

def install_kenlm():
    kenlm_package="https://github.com/kpu/kenlm/archive/master.zip"
    subprocess.call([sys.executable, '-m', 'pip', 'install', '{0}'.format(kenlm_package)])

try:
    import kenlm
except ImportError as error:
    print("can't import kenlm")
    print("Installing kenlm")
    install_kenlm()
except Exception as exception:
    # Output unexpected Exceptions.
    print(exception, False)
    print(exception.__class__.__name__ + ": " + exception.message)

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
    install_requires=[
        "numpy>=1.16",
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)

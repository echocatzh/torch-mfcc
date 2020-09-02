from setuptools import find_packages, setup

NAME = 'torch_mfcc'
VERSION = "0.1.3"
REQUIREMENTS = [
    'numpy',
    'scipy',
    'librosa'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name=NAME,
      version=VERSION,
      description="A librosa's STFT/FBANK/MFCC implement based on Torch",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/echocatzh/torch-mfcc",
      author="Shimin Zhang",
      author_email="shmzhang@npu-aslp.org",
      packages=["torch_mfcc"],
      install_requires=REQUIREMENTS,
    #   python_requires=">=3.5",
      license="MIT")
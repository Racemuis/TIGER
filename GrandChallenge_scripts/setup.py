from setuptools import setup, find_packages

setup(
    name="first_submission",
    version="0.0.1",
    author="T-ISMI",
    author_email="chiara.thoeni@ru.nl",
    packages=find_packages(),
    license="LICENSE.txt",
    install_requires=[
        "protobuf == 3.20.1",
        "numpy==1.20.2",
        "tqdm==4.62.3",
        "tensorflow==2.8.0",
        "opencv-python==4.5.5.64",
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "Pillow>=7.1.2",
        "matplotlib>=3.2.2",
        "pandas>=1.1.4",
        "seaborn>=0.11.0",
        "tensorboard>=2.4.1",
        "PyYAML>=5.3.1",
        "requests>=2.23.0",
        "scipy>=1.4.1"
    ],
)

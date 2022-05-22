from setuptools import setup, find_packages

setup(
    name="first_submission",
    version="0.0.1",
    author="T-ISMI",
    author_email="chiara.thoeni@ru.nl",
    packages=find_packages(),
    license="LICENSE.txt",
    install_requires=[
        "numpy==1.20.2",
        "tqdm==4.62.3",
        "tensorflow==2.8.0",
        "opencv-python==4.5.5.64"
    ],
)

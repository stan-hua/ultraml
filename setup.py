from setuptools import setup, find_packages

setup(
    name="ultraml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fire",
        "opencv-python",
        "numpy",
        "scikit-learn",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "ultraml=ultraml.core.extract:main",
        ],
    },
    author="Stanley Bryan Zamora Hua",
    author_email="stanley.z.hua@gmail.com",
    description="A simple package for preprocessing ultrasound imaging data for machine learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/stan-hua/ultraml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

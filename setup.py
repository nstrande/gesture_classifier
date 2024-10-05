from setuptools import setup, find_packages

setup(
    name="gesture_classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="Niels Strande",
    description="A simple gesture classification project",
    long_description=open("README.md").read(),
    url="TBD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

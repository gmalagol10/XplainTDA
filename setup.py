from setuptools import setup, find_packages

setup(
    name="xplaintda",
    version="0.1",
    packages=find_packages(),
    install_requires=[l.strip() for l in open('requirements.txt').readlines()],
    author="Gabriele",
    author_email="gabriele.malagoli3@gmail.com",
    description="XplainTDA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gmalagol10/xplaintda.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)

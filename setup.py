from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="tensor_track",
    version="0.0.1",
    description="The one-stop shop for tracking TensorFlow metrics",
    long_description=long_description,
    long_desctiption_contant_type="text/markdown",
    url="https://github.com/SamClaflin/Tensor-Track",
    author="Sam Claflin",
    author_email="samclaflin7@gmail.com",
    license="MIT",
    packages=["tensor_track"],
    zip_safe=False
)

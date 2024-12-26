from setuptools import setup

from setuptools import setup, find_packages

setup(
    name="scirex",
    version="1.0.0",
    packages=find_packages(where="scirex"),
    package_dir={"": "scirex"},
)
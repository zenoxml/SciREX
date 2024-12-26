from setuptools import setup, find_packages

setup(
    name="scirex",
    version="1.0.0",
    packages=find_packages(),  # Remove the 'where' parameter
    python_requires='>=3.10',  # Match your pyproject.toml requirement
)
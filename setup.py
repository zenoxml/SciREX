# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and 
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org

import os
from setuptools import setup, find_packages

# Read version from __init__.py
def get_version():
    init_path = os.path.join("src", "scirex", "__init__.py")
    with open(init_path) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Version not found")

# Read long description from README
def get_long_description():
    with open("README.md", encoding="utf-8") as f:
        return f.read()

# Core dependencies
REQUIREMENTS = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "torch>=2.0.0",
    "matplotlib>=3.7.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.2.0",
    "plotly>=5.13.0",
    "pyyaml>=6.0.0",
]

# Development dependencies
DEV_REQUIREMENTS = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.3.0",
    "pytest-benchmark>=4.0.0",
    "pytest-timeout>=2.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "sphinx>=6.0.0",
    "sphinx-rtd-theme>=1.2.0",
]

# Research dependencies
RESEARCH_REQUIREMENTS = [
    "jupyter>=1.0.0",
    "jupyterlab>=4.0.0",
    "seaborn>=0.12.0",
    "tensorboard>=2.12.0",
    "wandb>=0.15.0",
]

# CUDA dependencies
CUDA_REQUIREMENTS = [
    "cupy>=12.0.0",
    "torch>=2.0.0+cu118",
]

setup(
    name="scirex",
    version=get_version(),
    description="Scientific Research and Engineering eXcellence Framework",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Zenteiq Aitech Innovations and AiREX Lab",
    author_email="contact@scirex.org",
    maintainer="Zenteiq Aitech Innovations",
    maintainer_email="sashi@scirex.org",
    url="https://scirex.org",
    project_urls={
        "Documentation": "https://scirex.org/docs",
        "Source": "https://github.com/zenoxml/SciREX",
        "Tracker": "https://github.com/zenoxml/SciREX/issues",
        "Discord": "https://discord.gg/NWcCPx22Hq",
    },
    license="Apache License 2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "research": RESEARCH_REQUIREMENTS,
        "cuda": CUDA_REQUIREMENTS,
        "all": DEV_REQUIREMENTS + RESEARCH_REQUIREMENTS + CUDA_REQUIREMENTS,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    entry_points={
        "console_scripts": [
            "scirex=scirex.cli:main",
        ],
    },
    zip_safe=False,
    keywords=[
        "scientific-computing",
        "machine-learning",
        "physics-informed-neural-networks",
        "differential-equations",
        "scientific-visualization",
    ],
    platforms=["any"],
    test_suite="tests",
)

# Post-installation message
if "install" in os.sys.argv or "develop" in os.sys.argv:
    print("""
    Thank you for installing SciREX!
    
    Quick start:
    - Documentation: https://scirex.org/docs
    - Tutorials: https://scirex.org/docs/tutorials
    - Examples: https://scirex.org/docs/examples
    
    For GPU support, install additional dependencies:
        pip install scirex[cuda]
    
    For development tools:
        pip install scirex[dev]
    
    For research tools:
        pip install scirex[research]
    
    Join our community:
    - Discord: https://discord.gg/scirex
    """)

├── CONTRIBUTORS.md
├── COPYRIGHT.md
├── CopyrightHeader.txt
├── LICENSE
├── PLATFORM.md
├── README.md
├── data
│   ├── README.md
│   └── synthetic
│       └── pdw-simulator
│           └── __init__.py
├── docs
│   ├── conf.py
│   ├── index.rst
│   └── tutorials
│       ├── advanced_usage.md
│       └── getting_started.md
├── experiments
│   ├── configs
│   │   └── default.yaml
│   └── results
│       └── benchmarks
│           └── __init__.py
├── pyproject.toml
├── pytest.ini
├── requirements
│   ├── base.txt
│   ├── dev.txt
│   └── research.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── dl
│   │   │   └── __init__.py
│   │   ├── ml
│   │   │   ├── __init__.py
│   │   │   ├── supervised
│   │   │   │   ├── __init__.py
│   │   │   │   ├── classification
│   │   │   │   │   └── __init__.py
│   │   │   │   └── regression
│   │   │   │       └── __init__.py
│   │   │   └── unsupervised
│   │   │       ├── __init__.py
│   │   │       ├── clustering
│   │   │       │   └── __init__.py
│   │   │       ├── dimensionality_reduction
│   │   │       │   └── __init__.py
│   │   │       └── feature_selection
│   │   │           └── __init__.py
│   │   ├── postprocessing
│   │   │   └── __init__.py
│   │   └── sciml
│   │       ├── __init__.py
│   │       ├── super_resolution
│   │       │   └── __init__.py
│   │       └── vpinns
│   │           └── __init__.py
│   └── hardware
│       └── __init__.py
├── tests
│   └── __init__.py
└── tools
    ├── coverage
    ├── lint
    └── typing
        └── mypy.ini

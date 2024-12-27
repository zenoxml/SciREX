├── CONTRIBUTING.md
├── CONTRIBUTORS.md
├── CopyrightHeader.txt
├── LICENSE
├── PLATFORM.md
├── README.md
├── examples
│   └── sciml
│       └── fastvpinns
│           └── forward_problems
│               ├── README.md
│               └── example_poisson2d.py
├── pyproject.toml
├── pytest.ini
├── requirements
│   ├── all.txt
│   └── fastvpinns.txt
├── scirex
│   ├── __init__.py
│   └── core
│       ├── __init__.py
│       └── sciml
│           ├── fe
│           │   ├── FE2D_Cell.py
│           │   ├── README.md
│           │   ├── __init__.py
│           │   ├── basis_2d_qn_chebyshev_2.py
│           │   ├── basis_2d_qn_jacobi.py
│           │   ├── basis_2d_qn_legendre.py
│           │   ├── basis_2d_qn_legendre_special.py
│           │   ├── basis_function_2d.py
│           │   ├── basis_function_3d.py
│           │   ├── fe2d_setup_main.py
│           │   ├── fe_transformation_2d.py
│           │   ├── fe_transformation_3d.py
│           │   ├── fespace.py
│           │   ├── fespace2d.py
│           │   ├── quad_affine.py
│           │   ├── quad_bilinear.py
│           │   ├── quadratureformulas.py
│           │   └── quadratureformulas_quad2d.py
│           ├── Geometry
│           │   ├── __init__.py
│           │   ├── geometry.py
│           │   └── geometry_2d.py
│           ├── __init__.py
│           ├── fastvpinns
│           │   ├── __init__.py
│           │   ├── data
│           │   │   ├── __init__.py
│           │   │   ├── datahandler.py
│           │   │   └── datahandler2d.py
│           │   ├── model
│           │   │   ├── __init__.py
│           │   │   ├── model.py
│           │   │   ├── model_hard.py
│           │   │   ├── model_inverse.py
│           │   │   └── model_inverse_domain.py
│           │   └── physics
│           │       ├── __init__.py
│           │       ├── cd2d.py
│           │       ├── cd2d_inverse.py
│           │       ├── cd2d_inverse_domain.py
│           │       ├── helmholtz2d.py
│           │       ├── poisson2d.py
│           │       └── poisson2d_inverse.py
│           └── utils
│               ├── __init__.py
│               ├── compute_utils.py
│               ├── plot_utils.py
│               └── print_utils.py
├── setup.py
└── tests
    ├── __init__.py
    ├── src
    │   └── core
    │       └── sciml
    │           └── fastvpinns
    │               └── unit
    │                   └── FE_2D
    │                       ├── test_fe2d_cell.py
    │                       ├── test_fespace2d.py
    │                       ├── test_quadrature.py
    │                       └── test_quadratureformulas_quad2d.py
    └── support_files
        ├── circle_quad.mesh
        ├── circle_quad.txt
        ├── circle_quad_wrong.mesh
        ├── const_inverse_poisson_solution.txt
        └── fem_output_circle2.csv

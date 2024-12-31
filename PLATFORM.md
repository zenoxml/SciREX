├── CONTRIBUTING.md
├── CONTRIBUTORS.md
├── CopyrightHeader.txt
├── LICENSE
├── PLATFORM.md
├── README.md
├── docs
│   ├── api
│   │   └── core
│   │       ├── dl
│   │       │   └── tensorflow_wrapper.md
│   │       ├── ml
│   │       │   └── unsupervised
│   │       │       └── clustering
│   │       │           ├── agglomerative.md
│   │       │           ├── base.md
│   │       │           ├── dbscan.md
│   │       │           ├── gmm.md
│   │       │           ├── hdbscan.md
│   │       │           ├── kmeans.md
│   │       │           └── optics.md
│   │       ├── postprocessing
│   │       │   └── postprocessing.md
│   │       ├── sciml
│   │       │   ├── fastvpinns
│   │       │   │   ├── data
│   │       │   │   │   ├── datahandler.md
│   │       │   │   │   └── datahandler2d.md
│   │       │   │   ├── model
│   │       │   │   │   ├── model.md
│   │       │   │   │   ├── model_hard.md
│   │       │   │   │   ├── model_inverse.md
│   │       │   │   │   └── model_inverse_domain.md
│   │       │   │   └── physics
│   │       │   │       ├── cd2d.md
│   │       │   │       ├── cd2d_inverse.md
│   │       │   │       ├── cd2d_inverse_domain.md
│   │       │   │       ├── helmholtz2d.md
│   │       │   │       ├── poisson2d.md
│   │       │   │       └── poisson_2d_inverse.md
│   │       │   ├── fe
│   │       │   │   ├── basis_2d_QN_Chebyshev_2.md
│   │       │   │   ├── basis_2d_QN_Jacobi.md
│   │       │   │   ├── basis_2d_QN_Legendre.md
│   │       │   │   ├── basis_2d_QN_Legendre_Special.md
│   │       │   │   ├── basis_function_2d.md
│   │       │   │   ├── basis_function_3d.md
│   │       │   │   ├── fe2d_cell.md
│   │       │   │   ├── fe2d_setup_main.md
│   │       │   │   ├── fe_transformation_2d.md
│   │       │   │   ├── fe_transformation_3d.md
│   │       │   │   ├── fespace.md
│   │       │   │   ├── fespace2d.md
│   │       │   │   ├── quad_affine.md
│   │       │   │   ├── quad_bilinear.md
│   │       │   │   ├── quadratureformulas.md
│   │       │   │   └── quadratureformulas_quad2d.md
│   │       │   └── geometry
│   │       │       ├── geometry.md
│   │       │       └── geometry_2d.md
│   │       └── tutorial
│   │           └── classification_tutorial.md
│   ├── assets
│   │   ├── 8d22b1ef-da29-4f21-b593-a5f3165ed838.png
│   │   ├── logo.png
│   │   └── scirex-logo.svg
│   ├── conf.py
│   ├── index.md
│   └── index.rst
├── examples
│   ├── dl
│   │   ├── cnn-mnist.py
│   │   └── fcnn-mnist.py
│   ├── ml
│   │   ├── supervised
│   │   │   └── classification
│   │   │       ├── README.md
│   │   │       ├── example_logistic_regression.py
│   │   │       └── example_naive_bayes.py
│   │   └── unsupervised
│   │       └── clustering
│   │           ├── README.md
│   │           └── example_kmeans.py
│   └── sciml
│       └── fastvpinns
│           └── forward_problems
│               ├── example_helmholtz2d.md
│               ├── example_helmholtz2d.py
│               ├── example_poisson2d.md
│               └── example_poisson2d.py
├── mkdocs.yml
├── pyproject.toml
├── pytest.ini
├── requirements
│   ├── all.txt
│   └── fastvpinns.txt
├── scirex
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   └── core
│       ├── __init__.py
│       ├── __pycache__
│       │   └── __init__.cpython-311.pyc
│       ├── dl
│       │   ├── README.md
│       │   ├── __init__.py
│       │   ├── fcnn.py
│       │   └── tensorflow_wrapper.py
│       ├── ml
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   └── __init__.cpython-311.pyc
│       │   ├── supervised
│       │   │   ├── __init__.py
│       │   │   └── classification
│       │   │       ├── __init__.py
│       │   │       ├── base.py
│       │   │       ├── logistic_regression.py
│       │   │       └── naive_bayes.py
│       │   └── unsupervised
│       │       ├── __init__.py
│       │       ├── __pycache__
│       │       │   └── __init__.cpython-311.pyc
│       │       ├── clustering
│       │       │   ├── __init__.py
│       │       │   ├── __pycache__
│       │       │   │   ├── __init__.cpython-311.pyc
│       │       │   │   ├── agglomerative.cpython-311.pyc
│       │       │   │   ├── base.cpython-311.pyc
│       │       │   │   ├── dbscan.cpython-311.pyc
│       │       │   │   ├── gmm.cpython-311.pyc
│       │       │   │   ├── hdbscan.cpython-311.pyc
│       │       │   │   ├── kmeans.cpython-311.pyc
│       │       │   │   └── optics.cpython-311.pyc
│       │       │   ├── agglomerative.py
│       │       │   ├── base.py
│       │       │   ├── dbscan.py
│       │       │   ├── gmm.py
│       │       │   ├── hdbscan.py
│       │       │   ├── kmeans.py
│       │       │   └── optics.py
│       │       ├── dimensionality_reduction
│       │       │   └── __init__.py
│       │       └── feature_selection
│       │           └── __init__.py
│       └── sciml
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
│           ├── fe
│           │   ├── README.md
│           │   ├── __init__.py
│           │   ├── basis_2d_qn_chebyshev_2.py
│           │   ├── basis_2d_qn_jacobi.py
│           │   ├── basis_2d_qn_legendre.py
│           │   ├── basis_2d_qn_legendre_special.py
│           │   ├── basis_function_2d.py
│           │   ├── basis_function_3d.py
│           │   ├── fe2d_cell.py
│           │   ├── fe2d_setup_main.py
│           │   ├── fe_transformation_2d.py
│           │   ├── fe_transformation_3d.py
│           │   ├── fespace.py
│           │   ├── fespace2d.py
│           │   ├── quad_affine.py
│           │   ├── quad_bilinear.py
│           │   ├── quadratureformulas.py
│           │   └── quadratureformulas_quad2d.py
│           ├── geometry
│           │   ├── __init__.py
│           │   ├── geometry.py
│           │   └── geometry_2d.py
│           └── utils
│               ├── __init__.py
│               ├── compute_utils.py
│               ├── plot_utils.py
│               └── print_utils.py
├── setup.py
└── tests
    ├── __init__.py
    ├── __pycache__
    │   └── __init__.cpython-311.pyc
    ├── src
    │   └── core
    │       ├── dl
    │       │   └── test_fcnn.py
    │       ├── ml
    │       │   ├── classification
    │       │   │   ├── test_logistic_regression.py
    │       │   │   └── test_naive_bayes.py
    │       │   └── unsupervised
    │       │       └── clustering
    │       │           ├── __pycache__
    │       │           │   └── test_hdbscan.cpython-311-pytest-8.3.4.pyc
    │       │           ├── plots
    │       │           ├── test_agglomerative.py
    │       │           ├── test_dbscan.py
    │       │           ├── test_gmm.py
    │       │           ├── test_hdbscan.py
    │       │           └── test_kmeans.py
    │       └── sciml
    │           └── fastvpinns
    │               └── unit
    │                   ├── data
    │                   │   └── test_data_functions.py
    │                   ├── fe_2d
    │                   │   ├── test_fe2d_cell.py
    │                   │   ├── test_fespace2d.py
    │                   │   ├── test_quadrature.py
    │                   │   └── test_quadratureformulas_quad2d.py
    │                   └── geometry
    │                       ├── test_dirichlet_boundary.py
    │                       ├── test_geometry_2d.py
    │                       ├── test_numtest.py
    │                       ├── test_read_mesh.py
    │                       └── test_write_vtk.py
    └── support_files
        ├── chainlink.txt
        ├── circle_quad.mesh
        ├── circle_quad.txt
        ├── circle_quad_wrong.mesh
        ├── const_inverse_poisson_solution.txt
        ├── engytime.txt
        └── fem_output_circle2.csv

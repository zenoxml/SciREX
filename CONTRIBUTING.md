# Contributing to SciREX

First off, thank you for considering contributing to SciREX! We value contributions from both the academic and industrial communities.

All types of contributions are encouraged and valued. This document outlines various ways to help and details about how this project handles them. Please read the relevant sections before making your contribution.

> If you like the project but don't have time to contribute, there are other ways to support it:
> - Star the project on GitHub
> - Share it in academic and professional networks
> - Cite it in your research papers
> - Mention it at conferences and workshops
> - Use it in your research or industrial projects

## Table of Contents

- [Questions and Support](#questions-and-support)
- [Ways to Contribute](#ways-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Code Contributions](#code-contributions)
  - [Documentation](#documentation)
  - [Research Contributions](#research-contributions)
- [Development Guidelines](#development-guidelines)
  - [Setting Up Development Environment](#setting-up-development-environment)
  - [Code Style](#code-style)
  - [Testing](#testing)
  - [Documentation Standards](#documentation-standards)
- [Legal Notices](#legal-notices)

## Questions and Support

Before asking a question:
- Read the [Documentation](https://scirex.org/docs)
- Search existing [Issues](https://github.com/zenoxml/SciREX/issues)
- Check our [Discord community](https://discord.gg/NWcCPx22Hq/)

For new questions:
1. Open an [Issue](https://github.com/zenoxml/SciREX/issues/new)
2. Provide context about your use case
3. Include relevant details (Python version, OS, etc.)
4. Specify if it's a research or industrial application

## Ways to Contribute

### Reporting Bugs

#### Before Submitting a Bug Report

- Verify you're using the latest version
- Check if it's a known issue in our [bug tracker](https://github.com/zenoxml/SciREX/issues)
- Search academic literature if it's a known scientific computing issue
- Collect information:
  - Full error traceback
  - OS and hardware details
  - Minimal reproducible example
  - Scientific context if applicable

#### Submitting a Bug Report

> **Security Issues**: Never report security vulnerabilities through public GitHub issues. Email security@zenteiq.ai instead.

Submit bugs through GitHub issues:
1. Use the bug report template
2. Describe expected vs actual behavior
3. Provide reproduction steps
4. Include relevant scientific context
5. Attach minimal test data if needed

### Suggesting Enhancements

For enhancement suggestions:
1. Verify the functionality doesn't exist
2. Explain the scientific/technical motivation
3. Provide mathematical foundations if applicable
4. Describe implementation approach
5. Reference relevant research papers

### Code Contributions

#### First-time Contributors

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/<YOUR_USERNAME>/SciREX.git
cd SciREX
```
3. Create a branch:
```bash
git checkout -b feature/<descriptive-name>
```
4. Install dependencies:
```bash
pip install -e .
```

#### Development Workflow

1. Write tests first
2. Implement your changes
3. Add documentation
4. Run test suite:
```bash
pytest tests/
```
5. Run black formatter: 
```bash 
black filename
```
6. Create a pull request

### Documentation

Documentation contributions should:
- Follow scientific writing standards
- Include mathematical notation where appropriate
- Provide practical examples
- Reference relevant literature
- Follow our documentation style guide

### Research Contributions

For research-related contributions:
- Provide theoretical foundations
- Include mathematical proofs if applicable
- Add benchmark results
- Reference related work
- Follow scientific reproducibility guidelines

## Development Guidelines

### Code Style

- Follow PEP 8
- Use meaningful scientific variable names
- Document mathematical equations
- Include algorithm complexity analysis
- Add references to papers/methods

### Testing

- Write unit tests for new features
- Include numerical accuracy tests
- Add performance benchmarks
- Test edge cases
- Verify mathematical correctness

## Styleguides

### Code Style and Documentation

#### Google Style Guide Adaptation for Scientific Computing

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with specific adaptations for scientific computing:

##### Function and Variable Names
```python
# Good
def solve_navier_stokes(reynolds_number, initial_conditions):
    velocity_field = initialize_flow_field()
    pressure_gradient = compute_pressure_gradient()
    
# Bad
def solveNS(re, ic):
    v = init_flow()
    dp = comp_press()
```

##### Docstring Format
```python
def compute_finite_difference(
    function_values: np.ndarray, 
    dx: float, 
    order: int = 2
) -> np.ndarray:
    """Computes finite difference approximation of derivatives.

    Implements centered finite difference scheme of specified order for
    approximating derivatives of discrete data.

    Args:
        function_values: Array of function values at discrete points.
            Shape: (n_points,) or (n_points, n_dimensions)
        dx: Grid spacing (assumed uniform)
        order: Order of accuracy for finite difference scheme.
            Must be 2 or 4. Defaults to 2.

    Returns:
        np.ndarray: Approximated derivatives at each point.
            Same shape as input function_values.

    Raises:
        ValueError: If order is not 2 or 4
        ValueError: If function_values has less than order + 1 points

    Notes:
        Uses the following stencils:
        2nd order: f'(x) ≈ (f(x+h) - f(x-h))/(2h)
        4th order: f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h))/(12h)

    References:
        [1] LeVeque, R. J. (2007). Finite Difference Methods for Ordinary 
            and Partial Differential Equations.
    """
```

##### Module-Level Docstrings
```python
"""Solvers for partial differential equations using finite element methods.

This module implements various finite element solvers for elliptic, parabolic,
and hyperbolic PDEs. The implementation follows the formulation in Hughes (2000)
with extensions for stabilized methods.

Key classes:
    - FEMSolver: Base solver class
    - EllipticSolver: For elliptic PDEs
    - ParabolicSolver: For parabolic PDEs

Key functions:
    - assemble_stiffness_matrix: Assembles global stiffness matrix
    - solve_linear_system: Solves resulting linear system

Note that all solvers assume a well-posed problem with appropriate boundary
conditions.

References:
    [1] Hughes, T. J. R. (2000). The Finite Element Method: Linear Static and 
        Dynamic Finite Element Analysis.
"""
```

##### Class Docstrings
```python
class AdaptiveMesh:
    """A mesh that adapts based on solution features.
    
    This class implements adaptive mesh refinement/coarsening based on
    error estimates. The refinement strategy follows the procedure outlined
    in Bank et al. (1983).
    
    Attributes:
        nodes: Array of node coordinates, shape (n_nodes, dimension)
        elements: Array of element connectivity, shape (n_elements, nodes_per_element)
        error_estimator: Callable that computes error estimates
        refinement_threshold: Float, threshold for mesh refinement
        
    Example:
        >>> mesh = AdaptiveMesh(domain=[0, 1], initial_elements=10)
        >>> mesh.refine(error_threshold=1e-3)
        >>> solution = solve_pde(mesh)
    
    References:
        [1] Bank, R. E., Sherman, A. H., & Weiser, A. (1983). "Refinement 
            Algorithms and Data Structures for Regular Local Mesh Refinement"
    """
```

##### Exception Handling
```python
def integrate_numerically(function, bounds, method='gauss'):
    """Performs numerical integration using specified method.
    
    Args:
        function: Callable to integrate
        bounds: Tuple of (lower, upper) bounds
        method: Integration method ('gauss' or 'simpson')
        
    Raises:
        ValueError: If bounds are invalid or method is unsupported
        IntegrationError: If integration fails to converge
        
    Note:
        For singular integrands, consider using adaptive methods
    """
    if bounds[0] >= bounds[1]:
        raise ValueError(
            f"Lower bound {bounds[0]} must be less than upper bound {bounds[1]}"
        )
    
    try:
        result = _perform_integration(function, bounds, method)
    except ConvergenceError as e:
        raise IntegrationError(f"Integration failed to converge: {e}")
```

##### Type Hints and Comments
```python
from typing import Union, Callable, Optional
import numpy as np
import scipy.sparse as sp

def solve_sparse_system(
    matrix: Union[np.ndarray, sp.spmatrix],  # System matrix
    rhs: np.ndarray,  # Right-hand side vector
    solver: Optional[str] = 'gmres',  # Iterative solver type
    preconditioner: Optional[Callable] = None,  # Custom preconditioner
    tolerance: float = 1e-6
) -> np.ndarray:
    """Solves sparse linear system Ax = b."""

Example docstring:
```python
def solve_pde(mesh, boundary_conditions, tolerance=1e-6):
    """Solves partial differential equations using the finite element method.
    
    Implements the algorithm described in [Smith et al., 2023] using an adaptive
    mesh refinement strategy.
    
    The method solves the equation:
    
    ∂u/∂t = α∇²u + f(x,t)
    
    Args:
        mesh (FEMesh): Finite element mesh object
        boundary_conditions (dict): Dictionary of boundary conditions
        tolerance (float, optional): Convergence tolerance. Defaults to 1e-6
        
    Returns:
        np.ndarray: Solution vector on mesh nodes
        
    References:
        [1] Smith et al. (2023). Advanced PDE Solvers. J. Comp. Physics
    """
```

### Commit Messages

- Use clear, descriptive commit messages
- Follow the format: `<type>(<scope>): <description>`
- Types: feat, fix, docs, style, refactor, perf, test
- Include issue numbers if applicable

Example:
```
feat(pinns): implement physics-informed loss function

- Add custom loss terms for PDE constraints
- Implement automatic differentiation for physics terms
- Add tests for conservation laws
- Update documentation with mathematical formulation

Fixes #123
```

## Improving The Documentation

Documentation improvements are highly valued. Please follow these guidelines:

### Types of Documentation

1. **API Documentation**
   - Clear function/class descriptions
   - Mathematical foundations
   - Parameter explanations
   - Example usage
   - References to papers

2. **Tutorials**
   - Step-by-step guides
   - Practical examples
   - Clear explanations
   - Reproducible code

3. **Theoretical Documentation**
   - Mathematical background
   - Algorithm descriptions
   - Derivations
   - Limitations and assumptions

4. **Benchmark Documentation**
   - Performance metrics
   - Comparison with other methods
   - Hardware specifications
   - Test conditions

### Documentation Standards

- Use clear, scientific language
- Include mathematical notation where appropriate
- Provide practical examples
- Reference relevant literature
- Follow consistent formatting

## Contributing Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/SciREX.git
   cd SciREX
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -e ".[dev,test,docs]"
   pre-commit install
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Development Process**
   - Write tests first
   - Implement changes
   - Add documentation
   - Run test suite
   - Check code style
   - Update benchmarks if needed

5. **Review Checklist**
   - [ ] Tests pass
   - [ ] Documentation updated
   - [ ] Code style guidelines followed
   - [ ] Benchmarks updated (if applicable)
   - [ ] Mathematical correctness verified
   - [ ] Examples added

6. **Submit Pull Request**
   - Use PR template
   - Link related issues
   - Describe changes
   - Add benchmark results
   - Request review

7. **Review Process**
   - Address reviewer comments
   - Update documentation
   - Maintain mathematical rigor
   - Ensure reproducibility

## Legal Notices

- All contributions must be licensed under the Apache License 2.0
- Contributors must sign our Contributor License Agreement
- Maintain all copyright notices
- Include appropriate citations

For detailed documentation and guidelines, visit [https://scirex.org/docs/contributing](https://scirex.org/docs/contributing)

---

*This document is maintained by the SciREX team at Zenteiq Aitech Innovations and AiREX Lab.*

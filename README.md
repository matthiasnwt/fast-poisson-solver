# Fast Poisson Solver

Short description of your project, what it does, the motivation behind it, and its core features.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Contributing](#contributing)
5. [License](#license)
6. [Contact](#contact)

## Installation

This project is available on PyPI and can be installed with pip:

```bash
pip install your-package-name
```

## Usage
The code is designed to be very flexible on the input and can take lists, numpy arrays and pytorch tensors as input.
Currently only 2d domains are supported.
The code needs x and y coordinates for the pde domain and x and y coordinates for the boundary region.
The coordinates order does not matter.
For each pair of x and y coordinate the pde domains needs a value for the source function and the boundary domain one for the boundary value.
For each domain and boundary region, the pre-computational has to be done only ones.
Then unlimited different source functions and boundary condition values can be calculated.
### Basic

```python
from fast_poisson_solver import Solver
solver = Solver()
solver.precompute(x_pde, y_pde, x_bc, y_bc)
u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f1, u_bc1)
u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f2, u_bc2)
...
```

### Advanced


## Features



## Contributing

We use SemVer for versioning.


## License


## Contact
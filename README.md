# 🧮 Poisson Solver with Tpetra & MueLu

This project implements a parallel 2D Poisson solver using Trilinos libraries such as **Tpetra**, **MueLu**, and **Teuchos**. It supports both standard and generalized Poisson systems and uses modern solver strategies (CG or GMRES) with multigrid preconditioning.

## 🔧 Features

* Finite difference discretization of 2D Poisson equation
* Distributed vector and matrix initialization with `Tpetra`
* Multigrid preconditioning via `MueLu`
* Customizable right-hand side and coefficients
* Output to per-rank files
* Command-line options for configuration


## 🧮 Supported Systems

The solver can handle the following types of linear systems:

### 1. Standard Poisson Equation

$$
-\Delta \phi = f
$$

Discretized using a standard 5-point stencil on a uniform 2D grid with homogeneous Dirichlet boundary conditions. Constructed via:

```cpp
createPoissonMatrix(...);
```

### 2. Generalized Poisson Equation

$$
-\nabla \cdot (n \nabla \phi) = f
$$

Where $n(x, y)$ is a spatially varying coefficient (e.g., density). Discretized using a conservative finite difference scheme. Constructed via:

```cpp
createGeneralizedPoissonMatrix(...);
```

To solve this form, pass the `--generalized` flag during runtime.

---



## 📦 Dependencies

* C++11 or newer
* [MPI](https://www.open-mpi.org/)
* [Trilinos](https://trilinos.github.io/) (with `Teuchos`, `Tpetra`, `MueLu`)

---

## ⚙️ Building the Code

Set `Trilinos_DIR` in `CMakePresets.json` to your install path, then configure and build using a preset:

```bash
cmake --preset CPU    # or GPU for CUDA builds
cmake --build build
```

---

## 🚀 Running the Solver

```bash
mpirun -n 4 ./build/Poisson --nx=100 --ny=100 --solver=GMRES
```

### Available Options

| Option              | Description                     | Default |
| ------------------- | ------------------------------- | ------- |
| `--nx`              | Grid size in x-direction        | `10`    |
| `--ny`              | Grid size in y-direction        | `10`    |
| `--solver`          | Linear solver (`CG` or `GMRES`) | `GMRES` |
| `--test_analytical` | Use analytical test case        | `false` |
| `--generalized`     | Use generalized Poisson matrix  | `false` |

---

## 🗂 Output

Each MPI rank produces an output file containing:

* Initial guess (`phi0_rank.out`)
* Right-hand side (`rhs_rank.out`)
* Final solution (`phi_rank.out`)
* Optional: density (`n_rank.out`) for generalized case

Each file contains tuples in the format `(i, j, value)`.

---

## 📁 Directory Structure

```
.
├── src/                    # C++ source files
│   ├── main.cpp
│   ├── matrix.cpp / .hpp
│   ├── initialization.cpp / .hpp
│   └── hello_world.cpp     # standalone Trilinos hello-world example
├── plot/                   # Jupyter notebooks for analysis and visualization
│   ├── analyse.ipynb
│   └── test_initial_cond.ipynb
├── build/                  # compiled binaries (gitignored)
├── data/                   # solver output files, *.out (gitignored)
├── results/                # timing and benchmark text files (gitignored)
├── CMakeLists.txt
├── CMakePresets.json
└── README.md
```

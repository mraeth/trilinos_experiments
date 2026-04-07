# Poisson Solver with Tpetra & MueLu

Parallel 2D Poisson solver using **Trilinos 17** (Tpetra, MueLu, Belos, Teuchos) and **Kokkos 5**. Supports both standard and generalized Poisson systems with CG or GMRES + multigrid preconditioning.

## Supported Systems

### Standard Poisson Equation

$$-\Delta \phi = f$$

Discretized with a 5-point finite difference stencil on a uniform grid with homogeneous Dirichlet boundary conditions.

### Generalized Poisson Equation

$$-\nabla \cdot (n(x,y) \, \nabla \phi) = f$$

Where $n(x, y)$ is a spatially varying coefficient. Uses a conservative face-averaging scheme. Pass `--generalized` at runtime.

---

## Dependencies

| Dependency | Version |
|---|---|
| C++ | 20 |
| CMake | ≥ 3.27 |
| MPI | any |
| Kokkos | 5.0.2 |
| Trilinos | 17.0.0 (Teuchos, Tpetra, Belos, MueLu, Ifpack2, Amesos2, Zoltan2) |

---

## Installation

Kokkos and Trilinos are built locally inside the repo. Run once (takes ~2–4 hours):

```bash
cd trilinos_install
mkdir build && cd build
cmake ..
cmake --build .
```

This installs into:
- `trilinos_install/kokkos-install/`
- `trilinos_install/trilinos-install/`

Override the install locations:
```bash
cmake .. -DKOKKOS_INSTALL_PREFIX=/your/kokkos/path \
          -DTRILINOS_INSTALL_PREFIX=/your/trilinos/path
```

---

## Building the Solver

```bash
cmake --preset CPU-T17
cmake --build build -j$(nproc)
```

Available presets in `CMakePresets.json`:

| Preset | Description |
|---|---|
| `CPU-T17` | Local Trilinos 17 + Kokkos 5 (C++20) |
| `CPU` | External Trilinos 16 install |
| `GPU` | Trilinos 16 + CUDA (Leonardo cluster) |

---

## Running the Solver

```bash
mpirun -n 4 ./build/Poisson --nx=100 --ny=100 --solver=GMRES
```

### Options

| Option | Description | Default |
|---|---|---|
| `--nx` | Grid points in x | `10` |
| `--ny` | Grid points in y | `10` |
| `--solver` | `CG` or `GMRES` | `GMRES` |
| `--generalized` | Use generalized Poisson matrix | `false` |
| `--test_analytical` | Verify against known analytical solution | `false` |

### Analytical test

The generalized case has an exact solution for validation:
```bash
mpirun -n 4 ./build/Poisson --nx=50 --ny=50 --solver=GMRES --test_analytical --generalized
```

---

## Output

Simulation outputs are written to `data/` (gitignored). Each MPI rank writes:

| File | Contents |
|---|---|
| `phi0_<rank>.out` | Initial guess |
| `rhs_<rank>.out` | Right-hand side |
| `phi_<rank>.out` | Final solution |
| `n_<rank>.out` | Density field (generalized case only) |

Each file contains `(i, j, value)` tuples. Timing results are written to `results/`.

---

## Analysis

Jupyter notebooks in `plot/` read the output files and visualize solutions:

- `plot/analyse.ipynb` — convergence, error, and scaling analysis
- `plot/test_initial_cond.ipynb` — initial condition inspection

---

## Directory Structure

```
.
├── src/
│   ├── main.cpp            # solver driver, vector init, file I/O
│   ├── matrix.cpp / .hpp   # Poisson and generalized Poisson assembly
│   └── initialization.hpp  # Kokkos-portable field functors
├── plot/                   # Jupyter notebooks
├── trilinos_install/       # Kokkos + Trilinos build scripts
│   ├── CMakeLists.txt
│   ├── kokkos-install/     # (gitignored)
│   └── trilinos-install/   # (gitignored)
├── build/                  # compiled binaries (gitignored)
├── data/                   # solver output *.out files (gitignored)
├── results/                # timing results (gitignored)
├── CMakeLists.txt
├── CMakePresets.json
└── README.md
```

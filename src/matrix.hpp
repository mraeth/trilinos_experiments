#pragma once

#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using ScalarView   = Kokkos::View<double*,  ExecutionSpace>;
// 2D view: view(i,j) with LayoutLeft maps to flat index i + j*nx
using ScalarView2D = Kokkos::View<double**, Kokkos::LayoutLeft, ExecutionSpace>;

// Opaque handle to an assembled matrix + preconditioner.
// Build via PoissonSolver::buildMatrix() / buildGeneralizedMatrix().
// Pass to PoissonSolver::solve() — can be reused across multiple solves.
class PoissonMatrix {
public:
    PoissonMatrix();
    ~PoissonMatrix();
    PoissonMatrix(PoissonMatrix&&) noexcept;
    PoissonMatrix& operator=(PoissonMatrix&&) noexcept;
private:
    friend class PoissonSolver;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class PoissonSolver {
public:
    // Must be called before constructing any PoissonSolver, and its lifetime
    // must exceed all PoissonSolver instances.
    struct ScopeGuard {
        ScopeGuard(int& argc, char**& argv);
        ~ScopeGuard();
    };

    PoissonSolver(int nx, int ny);
    ~PoissonSolver();

    // Standard Poisson: -Δφ = rhs
    PoissonMatrix buildMatrix();

    // Generalized Poisson: -∇·(n ∇φ) = rhs
    PoissonMatrix buildGeneralizedMatrix(const ScalarView2D& n);

    void apply(const PoissonMatrix& mat, const ScalarView2D& x, ScalarView2D& y);

    void solve(const PoissonMatrix& mat, const ScalarView2D& rhs, ScalarView2D& x,
               std::string solverType = "GMRES");

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

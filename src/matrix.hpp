#pragma once

#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

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
    PoissonMatrix buildGeneralizedMatrix(
        const Kokkos::View<double**>& n);

    void apply(const PoissonMatrix& mat,
               const Kokkos::View<double**>& x,
               Kokkos::View<double**>& y);

    void solve(const PoissonMatrix& mat,
               const Kokkos::View<double**>& rhs,
               Kokkos::View<double**>& x,
               std::string solverType = "GMRES");

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#pragma once
#include "tpetra_types.hpp"
#include <vector>

// Geometric multigrid V-cycle preconditioner for 2D tensor-product grids.
//
// Builds a hierarchy by halving each grid dimension per level, with bilinear
// interpolation as the prolongation operator.  Coarse operators are formed via
// the Galerkin product  A_c = P^T A_f P, so no re-assembly is needed.
// A damped Jacobi smoother is used at every level.
//
// Implements Tpetra::Operator so it plugs directly into any Belos solver as a
// left preconditioner without any other change to the solve setup.
//
// Requirements:
//   - Single MPI rank (serial map, as produced by makeSerialMap).
//   - Grid dimensions nx, ny should satisfy nx = 2^k + 1 for standard dyadic
//     coarsening; non-dyadic grids are handled but coarsening is less uniform.
class GeometricMGOperator : public TpetraOperator {
public:
    // Af        : assembled fine-level matrix (must have row map of size nx*ny)
    // nx, ny    : grid dimensions including boundary nodes
    // preSweeps / postSweeps : damped Jacobi sweeps before/after coarse correction
    // omega     : Jacobi damping factor (0.8 is a good default for Poisson)
    // maxLevels : upper bound on hierarchy depth
    GeometricMGOperator(RCP<TpetraCrsMatrix> Af, int nx, int ny,
                        int preSweeps  = 2,
                        int postSweeps = 2,
                        double omega   = 0.8,
                        int maxLevels  = 10);

    void apply(const TpetraMultiVector& X,
               TpetraMultiVector&       Y,
               Teuchos::ETransp         mode  = Teuchos::NO_TRANS,
               Scalar                   alpha = Teuchos::ScalarTraits<Scalar>::one(),
               Scalar                   beta  = Teuchos::ScalarTraits<Scalar>::zero()) const override;

    RCP<const TpetraMap> getDomainMap() const override;
    RCP<const TpetraMap> getRangeMap()  const override;

private:
    struct Level {
        int nx = 0, ny = 0;
        RCP<const TpetraMap> map;
        RCP<TpetraCrsMatrix> A;
        RCP<TpetraVector>    dinv;  // element-wise inverse of diag(A) for Jacobi
        RCP<TpetraCrsMatrix> P;     // prolongation from level+1 (coarser) to this
                                    // level (finer); null at the coarsest level
    };

    std::vector<Level> levels_;
    int    preSweeps_, postSweeps_;
    double omega_;

    void buildHierarchy(RCP<TpetraCrsMatrix> Af, int nx, int ny, int maxLevels);

    // Build the bilinear prolongation P : coarse (nxc×nyc) → fine (nxf×nyf).
    static RCP<TpetraCrsMatrix> buildProlongation(
        const RCP<const TpetraMap>& coarseMap, int nxc, int nyc,
        const RCP<const TpetraMap>& fineMap,   int nxf, int nyf);

    static RCP<TpetraVector> buildDinv(const RCP<TpetraCrsMatrix>& A);

    void vcycle(int lvl, TpetraVector& x, const TpetraVector& b) const;
    void smooth(int lvl, TpetraVector& x, const TpetraVector& b, int nSweeps) const;
};

#include "gmg.hpp"
#include <TpetraExt_MatrixMatrix.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

// ============================================================
//  Constructor / hierarchy setup
// ============================================================

GeometricMGOperator::GeometricMGOperator(
    RCP<TpetraCrsMatrix> Af, int nx, int ny,
    int preSweeps, int postSweeps, double omega, int maxLevels)
    : preSweeps_(preSweeps), postSweeps_(postSweeps), omega_(omega)
{
    buildHierarchy(Af, nx, ny, maxLevels);
}

// Build the multigrid hierarchy level by level, stopping when the coarser
// grid would have fewer than 3 nodes in some direction (i.e. only 1 interior
// node) or when maxLevels is reached.
void GeometricMGOperator::buildHierarchy(
    RCP<TpetraCrsMatrix> Af, int nx, int ny, int maxLevels)
{
    // Level 0 — finest
    {
        Level L;
        L.nx   = nx;  L.ny = ny;
        L.map  = Af->getRowMap();
        L.A    = Af;
        L.dinv = buildDinv(Af);
        L.P    = Teuchos::null;
        levels_.push_back(std::move(L));
    }

    while (static_cast<int>(levels_.size()) < maxLevels) {
        const Level& prev = levels_.back();
        const int nxc = (prev.nx + 1) / 2;
        const int nyc = (prev.ny + 1) / 2;

        // Stop if coarser grid has fewer than 3 nodes in any direction
        // (that means only 1 interior node — trivial coarse solve).
        if (std::min(nxc, nyc) < 3) break;

        auto comm = prev.map->getComm();
        auto cmap = rcp(new TpetraMap(
            static_cast<Tpetra::global_size_t>(nxc) * nyc, 0, comm));

        // Bilinear prolongation P : coarse → fine
        auto P = buildProlongation(cmap, nxc, nyc, prev.map, prev.nx, prev.ny);

        // Store P at the current (fine) level
        levels_.back().P = P;

        // Galerkin coarse operator:  Ac = P^T * A_f * P
        auto AP = rcp(new TpetraCrsMatrix(prev.map, 0));
        Tpetra::MatrixMatrix::Multiply(*prev.A, false, *P,  false, *AP);

        auto Ac = rcp(new TpetraCrsMatrix(P->getDomainMap(), 0));
        Tpetra::MatrixMatrix::Multiply(*P,      true,  *AP, false, *Ac);

        Level Lc;
        Lc.nx   = nxc;  Lc.ny = nyc;
        Lc.map  = cmap;
        Lc.A    = Ac;
        Lc.dinv = buildDinv(Ac);
        Lc.P    = Teuchos::null;
        levels_.push_back(std::move(Lc));
    }
}

// ============================================================
//  Tpetra::Operator interface
// ============================================================

RCP<const TpetraMap> GeometricMGOperator::getDomainMap() const {
    return levels_[0].A->getDomainMap();
}
RCP<const TpetraMap> GeometricMGOperator::getRangeMap() const {
    return levels_[0].A->getRangeMap();
}

// Standard preconditioner application:  Y = alpha * M^{-1} * X + beta * Y.
// Belos calls this with alpha=1, beta=0.
void GeometricMGOperator::apply(
    const TpetraMultiVector& X, TpetraMultiVector& Y,
    Teuchos::ETransp mode, Scalar alpha, Scalar beta) const
{
    TEUCHOS_TEST_FOR_EXCEPTION(
        mode != Teuchos::NO_TRANS, std::logic_error,
        "GeometricMGOperator::apply: transpose not supported");

    const int nv = static_cast<int>(X.getNumVectors());
    for (int v = 0; v < nv; ++v) {
        auto x_v = X.getVector(v);
        auto y_v = Y.getVectorNonConst(v);

        TpetraVector sol(levels_[0].map);
        sol.putScalar(0.0);
        vcycle(0, sol, *x_v);

        // y_v = alpha * sol + beta * y_v
        y_v->update(alpha, sol, beta);
    }
}

// ============================================================
//  Static helpers
// ============================================================

// Build the bilinear prolongation matrix from an (nxc × nyc) coarse grid to
// an (nxf × nyf) fine grid using standard dyadic coarsening rules.
//
// For fine node (i_f, j_f):
//   (even, even)  → coarse node injection,       weight 1
//   (odd,  even)  → x-edge midpoint interpolation, weights 1/2
//   (even, odd)   → y-edge midpoint interpolation, weights 1/2
//   (odd,  odd)   → cell-centre bilinear,          weights 1/4
//
// Fine boundary nodes that fall between coarse boundary nodes (only possible
// for non-dyadic grids) receive an empty row — their error stays 0 (Dirichlet).
RCP<TpetraCrsMatrix> GeometricMGOperator::buildProlongation(
    const RCP<const TpetraMap>& coarseMap, int nxc, int nyc,
    const RCP<const TpetraMap>& fineMap,   int nxf, int nyf)
{
    auto P = rcp(new TpetraCrsMatrix(fineMap, 4));

    auto add = [&](GlobalOrdinal kf, GlobalOrdinal kc, Scalar w) {
        Teuchos::Array<GlobalOrdinal> cols(1, kc);
        Teuchos::Array<Scalar>        vals(1, w);
        P->insertGlobalValues(kf, cols(), vals());
    };

    for (int jf = 0; jf < nyf; ++jf) {
        for (int if_ = 0; if_ < nxf; ++if_) {
            const GlobalOrdinal kf     = if_ + jf * nxf;
            const bool          i_even = (if_ % 2 == 0);
            const bool          j_even = (jf  % 2 == 0);
            const int           ic     = if_ / 2;
            const int           jc     = jf  / 2;

            if (i_even && j_even) {
                if (ic < nxc && jc < nyc)
                    add(kf, ic + jc * nxc, 1.0);

            } else if (!i_even && j_even) {
                if (ic + 1 < nxc && jc < nyc) {
                    add(kf,  ic      + jc * nxc, 0.5);
                    add(kf, (ic + 1) + jc * nxc, 0.5);
                }

            } else if (i_even && !j_even) {
                if (ic < nxc && jc + 1 < nyc) {
                    add(kf, ic + jc       * nxc, 0.5);
                    add(kf, ic + (jc + 1) * nxc, 0.5);
                }

            } else {  // !i_even && !j_even
                if (ic + 1 < nxc && jc + 1 < nyc) {
                    add(kf,  ic      + jc       * nxc, 0.25);
                    add(kf, (ic + 1) + jc       * nxc, 0.25);
                    add(kf,  ic      + (jc + 1) * nxc, 0.25);
                    add(kf, (ic + 1) + (jc + 1) * nxc, 0.25);
                }
            }
        }
    }

    // domainMap = coarse (input), rangeMap = fine (output)
    P->fillComplete(coarseMap, fineMap);
    return P;
}

// Returns a vector holding 1/diag(A), entry-wise. Zero diagonal entries
// (boundary identity rows may have value 1, so this is mainly a safety guard)
// are left as 0 — the smoother correction for that unknown will be zero, which
// is correct since Dirichlet rows need no update.
RCP<TpetraVector> GeometricMGOperator::buildDinv(const RCP<TpetraCrsMatrix>& A) {
    auto dinv = rcp(new TpetraVector(A->getRowMap()));
    A->getLocalDiagCopy(*dinv);
    auto view = dinv->getLocalViewHost(Tpetra::Access::ReadWrite);
    for (size_t i = 0; i < dinv->getLocalLength(); ++i) {
        const double d = view(i, 0);
        view(i, 0) = (std::abs(d) > 1e-15) ? 1.0 / d : 0.0;
    }
    return dinv;
}

// ============================================================
//  V-cycle
// ============================================================

void GeometricMGOperator::vcycle(
    int lvl, TpetraVector& x, const TpetraVector& b) const
{
    // Coarsest level: solve with many Jacobi sweeps (system is tiny, ≤ ~9 nodes).
    if (lvl == static_cast<int>(levels_.size()) - 1) {
        smooth(lvl, x, b, 50);
        return;
    }

    const Level& L  = levels_[lvl];
    const Level& Lc = levels_[lvl + 1];

    // Pre-smooth
    smooth(lvl, x, b, preSweeps_);

    // Residual:  r = b - A*x
    TpetraVector r(L.map);
    L.A->apply(x, r, Teuchos::NO_TRANS, -1.0, 0.0);  // r = -A*x
    r.update(1.0, b, 1.0);                            // r = b - A*x

    // Restrict to coarse level:  rc = P^T * r
    TpetraVector rc(Lc.map);
    L.P->apply(r, rc, Teuchos::TRANS);

    // Solve coarse correction:  A_c * ec = rc
    TpetraVector ec(Lc.map);
    ec.putScalar(0.0);
    vcycle(lvl + 1, ec, rc);

    // Prolongate correction:  x += P * ec
    L.P->apply(ec, x, Teuchos::NO_TRANS, 1.0, 1.0);

    // Post-smooth
    smooth(lvl, x, b, postSweeps_);
}

// Damped Jacobi:  x ← x + ω D⁻¹ (b − A x),  repeated nSweeps times.
void GeometricMGOperator::smooth(
    int lvl, TpetraVector& x, const TpetraVector& b, int nSweeps) const
{
    const Level& L = levels_[lvl];
    TpetraVector r(L.map);
    TpetraVector dr(L.map);

    for (int s = 0; s < nSweeps; ++s) {
        L.A->apply(x, r, Teuchos::NO_TRANS, -1.0, 0.0);  // r = -A*x
        r.update(1.0, b, 1.0);                            // r = b - A*x
        dr.elementWiseMultiply(1.0, *L.dinv, r, 0.0);     // dr = D^{-1} * r
        x.update(omega_, dr, 1.0);                         // x += ω * dr
    }
}

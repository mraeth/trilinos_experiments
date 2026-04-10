#include "matrix.hpp"
#include "tpetra_types.hpp"
#include "gmg.hpp"

struct PoissonMatrix::Impl {
    RCP<TpetraCrsMatrix> A;
    RCP<TpetraOperator>  prec;
};

struct PoissonSolver::Impl {
    RCP<const TpetraMapBase> map;
    int nx, ny;
};

RCP<const TpetraMapBase> makeSerialMap(int nx, int ny)
{
    const Tpetra::global_size_t n = static_cast<Tpetra::global_size_t>(nx) * ny;
    return rcp(new TpetraMapBase(n, 0, Teuchos::DefaultComm<int>::getComm()));
}


RCP<TpetraVector> wrapView(const RCP<const TpetraMapBase>& map, const ScalarView2D& v)
{
    using dev_view_2d = TpetraMultiVector::dual_view_type::t_dev;
    TpetraMultiVector mv(map, dev_view_2d(const_cast<double*>(v.data()), v.size(), 1));
    return rcp(new TpetraVector(mv, 0));
}

RCP<TpetraVector> wrapViewMut(const RCP<const TpetraMapBase>& map, ScalarView2D& v)
{
    using dev_view_2d = TpetraMultiVector::dual_view_type::t_dev;
    TpetraMultiVector mv(map, dev_view_2d(v.data(), v.size(), 1));
    return rcp(new TpetraVector(mv, 0));
}


PoissonSolver::ScopeGuard::ScopeGuard(int& argc, char**& argv) { Tpetra::initialize(&argc, &argv); }
PoissonSolver::ScopeGuard::~ScopeGuard()                       { Tpetra::finalize(); }

PoissonMatrix::PoissonMatrix() : impl_(std::make_unique<Impl>()) {}
PoissonMatrix::~PoissonMatrix() = default;
PoissonMatrix::PoissonMatrix(PoissonMatrix&&) noexcept = default;
PoissonMatrix& PoissonMatrix::operator=(PoissonMatrix&&) noexcept = default;

PoissonSolver::PoissonSolver(int nx, int ny)
    : impl_(std::make_unique<Impl>())
{
    impl_->nx  = nx;
    impl_->ny  = ny;
    impl_->map = makeSerialMap(nx, ny);
}

PoissonSolver::~PoissonSolver() = default;

PoissonMatrix PoissonSolver::buildMatrix()
{
    PoissonMatrix m;
    m.impl_->A    = createPoissonMatrix(impl_->map, impl_->nx, impl_->ny);
    m.impl_->prec = rcp(new GeometricMGOperator(m.impl_->A, impl_->nx, impl_->ny));
    return m;
}

PoissonMatrix PoissonSolver::buildGeneralizedMatrix(const ScalarView2D& n)
{
    PoissonMatrix m;
    m.impl_->A    = createGeneralizedPoissonMatrix(impl_->map, wrapView(impl_->map, n), impl_->nx, impl_->ny);
    m.impl_->prec = rcp(new GeometricMGOperator(m.impl_->A, impl_->nx, impl_->ny));
    return m;
}

PoissonMatrix PoissonSolver::buildHigherOrderGeneralizedMatrix(const ScalarView2D& n)
{
    PoissonMatrix m;
    m.impl_->A    = createHigherOrderGeneralizedPoissonMatrix(impl_->map, wrapView(impl_->map, n), impl_->nx, impl_->ny);
    m.impl_->prec = rcp(new GeometricMGOperator(m.impl_->A, impl_->nx, impl_->ny));
    return m;
}

void PoissonSolver::apply(const PoissonMatrix& mat, const ScalarView2D& x, ScalarView2D& y)
{
    mat.impl_->A->apply(*wrapView(impl_->map, x), *wrapViewMut(impl_->map, y));
}

void PoissonSolver::solve(const PoissonMatrix& mat, const ScalarView2D& rhs, ScalarView2D& x,
                          std::string solverType)
{
    auto b_tp = wrapView(impl_->map, rhs);
    auto x_tp = wrapViewMut(impl_->map, x);

    RCP<BelosLinearProblem> problem = rcp(new BelosLinearProblem());
    problem->setOperator(mat.impl_->A);
    problem->setLHS(x_tp);
    problem->setRHS(b_tp);
    problem->setLeftPrec(mat.impl_->prec);
    problem->setProblem();

    RCP<Teuchos::ParameterList> p = rcp(new Teuchos::ParameterList());
    p->set("Convergence Tolerance", 1e-10);
    p->set("Maximum Iterations", 500);
    p->set("Verbosity", Belos::Errors | Belos::Warnings | Belos::StatusTestDetails);
    p->set("Output Frequency", 1);

    Belos::SolverFactory<Scalar, TpetraMultiVector, TpetraOperator> factory;
    RCP<BelosSolverManager> solver = factory.create(solverType, p);
    solver->setProblem(problem);

    if (solver->solve() != Belos::Converged)
        std::cerr << "Warning: PoissonSolver did not converge." << std::endl;
}

void insertBoundaryRow(RCP<TpetraCrsMatrix>& A, GlobalOrdinal k)
{
    Teuchos::Array<GlobalOrdinal> cols(1, k);
    Teuchos::Array<Scalar>        vals(1, 1.0);
    A->insertGlobalValues(k, cols(), vals());
}

RCP<TpetraCrsMatrix> createPoissonMatrix(
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap, int nx, int ny)
{
    auto A = Teuchos::rcp(new TpetraCrsMatrix(rowMap, 5));

    const double stencil_scale = static_cast<double>(nx - 1) * static_cast<double>(ny - 1);

    for (GlobalOrdinal k = rowMap->getMinGlobalIndex(); k <= rowMap->getMaxGlobalIndex(); ++k) {
        const GlobalOrdinal i = k % nx;
        const GlobalOrdinal j = k / nx;

        Teuchos::Array<GlobalOrdinal> cols;
        Teuchos::Array<Scalar> vals;

        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            insertBoundaryRow(A, k);
            continue;
        }

        cols.push_back(k);      vals.push_back( 4.0 * stencil_scale);
        cols.push_back(k - 1);  vals.push_back(-1.0 * stencil_scale);
        cols.push_back(k + 1);  vals.push_back(-1.0 * stencil_scale);
        cols.push_back(k - nx); vals.push_back(-1.0 * stencil_scale);
        cols.push_back(k + nx); vals.push_back(-1.0 * stencil_scale);
        A->insertGlobalValues(k, cols(), vals());
    }

    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList());
    p->set("Optimize Storage", true);
    A->fillComplete(rowMap, rowMap, p);
    return A;
}

RCP<TpetraCrsMatrix> createGeneralizedPoissonMatrix(
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap,
    RCP<TpetraVector> n, int nx, int ny)
{
    auto n_view   = n->getLocalViewHost(Tpetra::Access::ReadOnly);
    auto A        = Teuchos::rcp(new TpetraCrsMatrix(rowMap, 5));
    const double stencil_scale = static_cast<double>(nx - 1) * static_cast<double>(ny - 1);

    for (GlobalOrdinal k = rowMap->getMinGlobalIndex(); k <= rowMap->getMaxGlobalIndex(); ++k) {
        const GlobalOrdinal i = k % nx;
        const GlobalOrdinal j = k / nx;

        Teuchos::Array<GlobalOrdinal> cols;
        Teuchos::Array<Scalar> vals;

        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            insertBoundaryRow(A, k);
            continue;
        }

        const Scalar nc = n_view(k, 0);
        const Scalar nE = (i < nx-1) ? (nc + n_view(k+1,  0)) / 2.0 : 0.0;
        const Scalar nW = (i > 0)    ? (nc + n_view(k-1,  0)) / 2.0 : 0.0;
        const Scalar nN = (j < ny-1) ? (nc + n_view(k+nx, 0)) / 2.0 : 0.0;
        const Scalar nS = (j > 0)    ? (nc + n_view(k-nx, 0)) / 2.0 : 0.0;

        if (i < nx-1) { cols.push_back(k+1);  vals.push_back(-nE * stencil_scale); }
        if (i > 0)    { cols.push_back(k-1);  vals.push_back(-nW * stencil_scale); }
        if (j < ny-1) { cols.push_back(k+nx); vals.push_back(-nN * stencil_scale); }
        if (j > 0)    { cols.push_back(k-nx); vals.push_back(-nS * stencil_scale); }
        cols.push_back(k); vals.push_back((nE + nW + nN + nS) * stencil_scale);

        A->insertGlobalValues(k, cols(), vals());
    }

    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList());
    p->set("Optimize Storage", true);
    A->fillComplete(rowMap, rowMap, p);
    return A;
}

// Higher-order (4th-order) generalized Poisson: -∇·(n ∇u)
//
// Discretizes the non-conservative form  -n u'' - n' u'  using 4th-order FD for every
// term.  The conservative and non-conservative forms are algebraically identical for
// smooth n; the non-conservative formulation avoids the fundamental problem with the
// conservative flux scheme, where computing (F_E - F_W)/h is inherently only 2nd-order
// accurate in the divergence even when the face fluxes are computed to higher order:
//
//   (F(x + h/2) - F(x - h/2))/h = F'(x) + h²/24 · F'''(x) + O(h⁴)
//
// Stencil for fully interior nodes (i ∈ [2, nx-3]):
//
//   u''_i = (-u_{i-2} + 16u_{i-1} - 30u_i + 16u_{i+1} - u_{i+2}) / (12 h²)
//   u'_i  = ( u_{i-2} -  8u_{i-1}         +  8u_{i+1} -  u_{i+2}) / (12 h)
//   n'_i  = ( n_{i-2} -  8n_{i-1}         +  8n_{i+1} -  n_{i+2}) / (12 h)
//
// Letting  A = n_i / (12 h²)  and  C = n'_i / (12 h) = n'_i · inv_h / 12:
//
//   coeff[i-2] =   A - C
//   coeff[i-1] = -16A + 8C
//   coeff[i]   =  30A
//   coeff[i+1] = -16A - 8C
//   coeff[i+2] =   A + C
//
// For i=1 and i=nx-2 the standard centered stencil reaches a phantom node outside the
// domain.  One-sided 4th-order Lagrange formulas are used instead (exact for polynomials
// up to degree 4, derived by requiring the stencil to reproduce 1, x, x², x³, x⁴):
//
//   i=1     forward:  u''(x_1) = ( 11u_0 - 20u_1 + 6u_2 + 4u_3 -  u_4) / (12h²)
//                     u'(x_1)  = ( -3u_0 - 10u_1 + 18u_2 - 6u_3 + u_4) / (12h)
//
//   i=n-2  backward:  u''(x_{n-2}) = (-u_{n-5} + 4u_{n-4} + 6u_{n-3} - 20u_{n-2} + 11u_{n-1}) / (12h²)
//                     u'(x_{n-2})  = (-u_{n-5} + 6u_{n-4} - 18u_{n-3} + 10u_{n-2} +  3u_{n-1}) / (12h)
//
// Requires nx, ny >= 5.
RCP<TpetraCrsMatrix> createHigherOrderGeneralizedPoissonMatrix(
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap,
    RCP<TpetraVector> n, int nx, int ny)
{
    auto n_view = n->getLocalViewHost(Tpetra::Access::ReadOnly);
    auto A = Teuchos::rcp(new TpetraCrsMatrix(rowMap, 9));

    const double inv_hx2 = static_cast<double>(nx - 1) * static_cast<double>(nx - 1);
    const double inv_hy2 = static_cast<double>(ny - 1) * static_cast<double>(ny - 1);

    for (GlobalOrdinal k = rowMap->getMinGlobalIndex(); k <= rowMap->getMaxGlobalIndex(); ++k) {
        const GlobalOrdinal i = k % nx;
        const GlobalOrdinal j = k / nx;

        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            insertBoundaryRow(A, k);
            continue;
        }

        // Accumulate stencil entries; shared indices (diagonal) are summed.
        std::map<GlobalOrdinal, Scalar> stencil;
        auto add = [&](GlobalOrdinal col, Scalar val) { stencil[col] += val; };

        const Scalar nc = n_view(k, 0);

        // --- X-direction: -n_i u''_i - (∂n/∂x)_i u'_i ---
        // All stencils cover 5 nodes; for n=const they reduce to the standard
        // 4th-order 5-point Laplacian  (-u_{i-2}+16u_{i-1}-30u_i+16u_{i+1}-u_{i+2})/(12h²).
        {
            const double Ax = nc * inv_hx2 / 12.0;
            if (i >= 2 && i <= nx - 3) {
                // Centered: uses k-2..k+2
                const double ndx = n_view(k-2,0) - 8.0*n_view(k-1,0) + 8.0*n_view(k+1,0) - n_view(k+2,0);
                const double Cx = ndx * inv_hx2 / 144.0;   // (∂n/∂x)_i · inv_hx / 12
                add(k-2,   Ax -      Cx);
                add(k-1, -16.0*Ax + 8.0*Cx);
                add(k,    30.0*Ax          );
                add(k+1, -16.0*Ax - 8.0*Cx);
                add(k+2,   Ax +      Cx);
            } else if (i == 1) {
                // Forward one-sided: uses k-1..k+3  (k-1 = left boundary, u=0 there)
                const double ndx = -3.0*n_view(k-1,0) - 10.0*nc + 18.0*n_view(k+1,0) - 6.0*n_view(k+2,0) + n_view(k+3,0);
                const double Cx = ndx * inv_hx2 / 144.0;
                add(k-1, -11.0*Ax +  3.0*Cx);
                add(k,    20.0*Ax + 10.0*Cx);
                add(k+1,  -6.0*Ax - 18.0*Cx);
                add(k+2,  -4.0*Ax +  6.0*Cx);
                add(k+3,   1.0*Ax -  1.0*Cx);
            } else { // i == nx-2
                // Backward one-sided: uses k-3..k+1  (k+1 = right boundary, u=0 there)
                const double ndx = -n_view(k-3,0) + 6.0*n_view(k-2,0) - 18.0*n_view(k-1,0) + 10.0*nc + 3.0*n_view(k+1,0);
                const double Cx = ndx * inv_hx2 / 144.0;
                add(k-3,   1.0*Ax +  1.0*Cx);
                add(k-2,  -4.0*Ax -  6.0*Cx);
                add(k-1,  -6.0*Ax + 18.0*Cx);
                add(k,    20.0*Ax - 10.0*Cx);
                add(k+1, -11.0*Ax -  3.0*Cx);
            }
        }

        // --- Y-direction: -n_i u''_y,i - (∂n/∂y)_i u'_y,i ---
        {
            const double Ay = nc * inv_hy2 / 12.0;
            if (j >= 2 && j <= ny - 3) {
                const double ndy = n_view(k-2*nx,0) - 8.0*n_view(k-nx,0) + 8.0*n_view(k+nx,0) - n_view(k+2*nx,0);
                const double Cy = ndy * inv_hy2 / 144.0;
                add(k-2*nx,  Ay -      Cy);
                add(k-nx,  -16.0*Ay + 8.0*Cy);
                add(k,      30.0*Ay          );
                add(k+nx,  -16.0*Ay - 8.0*Cy);
                add(k+2*nx,  Ay +      Cy);
            } else if (j == 1) {
                const double ndy = -3.0*n_view(k-nx,0) - 10.0*nc + 18.0*n_view(k+nx,0) - 6.0*n_view(k+2*nx,0) + n_view(k+3*nx,0);
                const double Cy = ndy * inv_hy2 / 144.0;
                add(k-nx,    -11.0*Ay +  3.0*Cy);
                add(k,        20.0*Ay + 10.0*Cy);
                add(k+nx,     -6.0*Ay - 18.0*Cy);
                add(k+2*nx,   -4.0*Ay +  6.0*Cy);
                add(k+3*nx,    1.0*Ay -  1.0*Cy);
            } else { // j == ny-2
                const double ndy = -n_view(k-3*nx,0) + 6.0*n_view(k-2*nx,0) - 18.0*n_view(k-nx,0) + 10.0*nc + 3.0*n_view(k+nx,0);
                const double Cy = ndy * inv_hy2 / 144.0;
                add(k-3*nx,   1.0*Ay +  1.0*Cy);
                add(k-2*nx,  -4.0*Ay -  6.0*Cy);
                add(k-nx,    -6.0*Ay + 18.0*Cy);
                add(k,       20.0*Ay - 10.0*Cy);
                add(k+nx,   -11.0*Ay -  3.0*Cy);
            }
        }

        Teuchos::Array<GlobalOrdinal> cols;
        Teuchos::Array<Scalar>        vals;
        for (auto& [col, val] : stencil) {
            cols.push_back(col);
            vals.push_back(val);
        }
        A->insertGlobalValues(k, cols(), vals());
    }

    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList());
    p->set("Optimize Storage", true);
    A->fillComplete(rowMap, rowMap, p);
    return A;
}

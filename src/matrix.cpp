#include "matrix.hpp"
#include "tpetra_types.hpp"

struct PoissonMatrix::Impl {
    RCP<TpetraCrsMatrix> A;
    RCP<TpetraOperator>  prec;
};

struct PoissonSolver::Impl {
    RCP<const TpetraMapBase> map;
    int nx, ny;
};

static RCP<const TpetraMapBase> makeSerialMap(int nx, int ny)
{
    const Tpetra::global_size_t n = static_cast<Tpetra::global_size_t>(nx) * ny;
    return rcp(new TpetraMapBase(n, 0, Teuchos::DefaultComm<int>::getComm()));
}

static RCP<TpetraOperator> buildPreconditioner(RCP<TpetraCrsMatrix> A)
{
    RCP<Teuchos::ParameterList> p = rcp(new Teuchos::ParameterList());
    p->set("verbosity", "low");
    p->set("coarse: max size", 32);
    p->set("cycle type", "V");
    p->set("max levels", 10);
    p->set("coarse: type", "Amesos2");

    for (int level = 0; level < 9; ++level) {
        Teuchos::ParameterList& lp = p->sublist("level " + std::to_string(level));
        lp.set("smoother: type", "RELAXATION");
        Teuchos::ParameterList sp;
        sp.set("relaxation: type", "Jacobi");
        sp.set("relaxation: damping factor", 0.8);
        sp.set("relaxation: sweeps", 2);
        lp.set("smoother: params", sp);
    }

    return MueLu::CreateTpetraPreconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A, *p);
}

static RCP<TpetraVector> wrapView(const RCP<const TpetraMapBase>& map, const ScalarView2D& v)
{
    using dev_view_2d = TpetraMultiVector::dual_view_type::t_dev;
    TpetraMultiVector mv(map, dev_view_2d(const_cast<double*>(v.data()), v.size(), 1));
    return rcp(new TpetraVector(mv, 0));
}

static void runBelos(RCP<TpetraCrsMatrix> A, RCP<TpetraOperator> prec,
                     RCP<TpetraVector> b, RCP<TpetraVector> x,
                     const std::string& solverType)
{
    RCP<BelosLinearProblem> problem = rcp(new BelosLinearProblem());
    problem->setOperator(A);
    problem->setLHS(x);
    problem->setRHS(b);
    problem->setLeftPrec(prec);
    problem->setProblem();

    RCP<Teuchos::ParameterList> p = rcp(new Teuchos::ParameterList());
    p->set("Convergence Tolerance", 1e-12);
    p->set("Maximum Iterations", 500);
    p->set("Verbosity", Belos::Errors | Belos::Warnings | Belos::StatusTestDetails);
    p->set("Output Frequency", 1);

    Belos::SolverFactory<Scalar, TpetraMultiVector, TpetraOperator> factory;
    RCP<BelosSolverManager> solver = factory.create(solverType, p);
    solver->setProblem(problem);

    if (solver->solve() != Belos::Converged)
        std::cerr << "Warning: PoissonSolver did not converge." << std::endl;
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
    m.impl_->prec = buildPreconditioner(m.impl_->A);
    return m;
}

PoissonMatrix PoissonSolver::buildGeneralizedMatrix(const ScalarView2D& n)
{
    PoissonMatrix m;
    m.impl_->A    = createGeneralizedPoissonMatrix(impl_->map, wrapView(impl_->map, n), impl_->nx, impl_->ny);
    m.impl_->prec = buildPreconditioner(m.impl_->A);
    return m;
}

void PoissonSolver::apply(const PoissonMatrix& mat, const ScalarView2D& x, ScalarView2D& y)
{
    mat.impl_->A->apply(*wrapView(impl_->map, x), *wrapView(impl_->map, y));
}

void PoissonSolver::solve(const PoissonMatrix& mat, const ScalarView2D& rhs, ScalarView2D& x,
                          std::string solverType)
{
    runBelos(mat.impl_->A, mat.impl_->prec,
             wrapView(impl_->map, rhs), wrapView(impl_->map, x), solverType);
}

RCP<TpetraCrsMatrix> createPoissonMatrix(
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap, int nx, int ny)
{
    auto A = Teuchos::rcp(new TpetraCrsMatrix(rowMap, 5));

    const double h_sq_inv = static_cast<double>(nx - 1) * static_cast<double>(ny - 1);

    for (GlobalOrdinal k = rowMap->getMinGlobalIndex(); k <= rowMap->getMaxGlobalIndex(); ++k) {
        const GlobalOrdinal i = k % nx;
        const GlobalOrdinal j = k / nx;

        Teuchos::Array<GlobalOrdinal> cols;
        Teuchos::Array<Scalar> vals;

        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            cols.push_back(k); vals.push_back(1.0);
        } else {
            cols.push_back(k);      vals.push_back( 4.0 * h_sq_inv);
            cols.push_back(k - 1);  vals.push_back(-1.0 * h_sq_inv);
            cols.push_back(k + 1);  vals.push_back(-1.0 * h_sq_inv);
            cols.push_back(k - nx); vals.push_back(-1.0 * h_sq_inv);
            cols.push_back(k + nx); vals.push_back(-1.0 * h_sq_inv);
        }
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
    const double h_sq_inv = static_cast<double>(nx - 1) * static_cast<double>(ny - 1);

    for (GlobalOrdinal k = rowMap->getMinGlobalIndex(); k <= rowMap->getMaxGlobalIndex(); ++k) {
        const GlobalOrdinal i = k % nx;
        const GlobalOrdinal j = k / nx;

        Teuchos::Array<GlobalOrdinal> cols;
        Teuchos::Array<Scalar> vals;

        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            cols.push_back(k); vals.push_back(1.0);
            A->insertGlobalValues(k, cols(), vals());
            continue;
        }

        const Scalar nc = n_view(k, 0);
        const Scalar nE = (i < nx-1) ? (nc + n_view(k+1,  0)) / 2.0 : 0.0;
        const Scalar nW = (i > 0)    ? (nc + n_view(k-1,  0)) / 2.0 : 0.0;
        const Scalar nN = (j < ny-1) ? (nc + n_view(k+nx, 0)) / 2.0 : 0.0;
        const Scalar nS = (j > 0)    ? (nc + n_view(k-nx, 0)) / 2.0 : 0.0;

        if (i < nx-1) { cols.push_back(k+1);  vals.push_back(-nE * h_sq_inv); }
        if (i > 0)    { cols.push_back(k-1);  vals.push_back(-nW * h_sq_inv); }
        if (j < ny-1) { cols.push_back(k+nx); vals.push_back(-nN * h_sq_inv); }
        if (j > 0)    { cols.push_back(k-nx); vals.push_back(-nS * h_sq_inv); }
        cols.push_back(k); vals.push_back((nE + nW + nN + nS) * h_sq_inv);

        A->insertGlobalValues(k, cols(), vals());
    }

    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList());
    p->set("Optimize Storage", true);
    A->fillComplete(rowMap, rowMap, p);
    return A;
}

#include <initialization.hpp>
#include <iostream>
#include <matrix.hpp>

#include <Kokkos_Core.hpp>

template<typename F>
concept GridFunctor = requires(F f, double x, double y) {
    { f(x, y) } -> std::convertible_to<double>;
};

template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node, GridFunctor CalculateFunc>
void initializeDistributed(
    const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> &rowMap,
    Teuchos::RCP<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> &b,
    GlobalOrdinal nx, GlobalOrdinal ny,
    CalculateFunc calculate_func) {


     std::cout << "Running on execution space in initialization: " << typeid(ExecutionSpace).name() << std::endl;

    b = Teuchos::rcp(new Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, true));
    const LocalOrdinal numLocalEntries = rowMap->getLocalNumElements();
    auto b_view = b->getLocalViewDevice(Tpetra::Access::OverwriteAll);

    Kokkos::parallel_for("InitializeVector", Kokkos::RangePolicy<ExecutionSpace>(0, numLocalEntries),
        KOKKOS_LAMBDA(const LocalOrdinal local_k) {
            const double x = static_cast<double>(local_k % nx) / static_cast<double>(nx - 1);
            const double y = static_cast<double>(local_k / nx) / static_cast<double>(ny - 1);
            b_view(local_k, 0) = calculate_func(x, y);
        });
}

template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
void print2File(const std::string &label,
                const Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &vec,
                GlobalOrdinal nx, GlobalOrdinal ny)
{
    std::ofstream outFile(label + ".out");

    if (!outFile.is_open()) {
        std::cerr << "Failed to open file " << label << ".out" << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(12);

    auto localView = vec.getLocalViewHost(Tpetra::Access::ReadOnly);
    auto data1d = Kokkos::subview(localView, Kokkos::ALL(), 0);

    const LocalOrdinal numLocal = vec.getMap()->getLocalNumElements();

    for (LocalOrdinal k = 0; k < numLocal; ++k) {
        outFile << "(" << k % nx << ", " << k / nx << ", " << data1d(k) << ")\n";
    }
}



RCP<TpetraVector> solve(RCP<TpetraCrsMatrix> A, RCP<TpetraVector> b, std::string solverType = "GMRES")
{
    using Teuchos::ParameterList;
  
    RCP<TpetraVector> x =
        rcp(new TpetraVector(b->getMap(), true));
RCP<Teuchos::ParameterList> mueluParams = rcp(new Teuchos::ParameterList());
mueluParams->set("verbosity", "low"); // Lower for production runs
mueluParams->set("coarse: max size", 32);
mueluParams->set("cycle type", "V"); // Consider F-cycle for speed

int maxLevels = 10;
mueluParams->set("max levels", maxLevels);

// Coarse grid solver
mueluParams->set("coarse: type", "Amesos2"); // Use a direct solver for the coarsest level

for (int level = 0; level < maxLevels - 1; ++level) {
    Teuchos::ParameterList& levelParams = mueluParams->sublist("level " + std::to_string(level));
    levelParams.set("smoother: type", "RELAXATION");
    Teuchos::ParameterList smootherParams;

    // Option 1: Damped Jacobi - often very good for GPUs
    smootherParams.set("relaxation: type", "Jacobi"); // Damped Jacobi
    smootherParams.set("relaxation: damping factor", 0.8);
    smootherParams.set("relaxation: sweeps", 2); // Few sweeps on fine levels

    // Option 2: Symmetric Gauss-Seidel - if coloring is efficient
    // smootherParams.set("relaxation: type", "Symmetric Gauss-Seidel");
    // smootherParams.set("relaxation: damping factor", 1.0); // No damping needed for SGS
    // smootherParams.set("relaxation: sweeps", (level < 3) ? 2 : (level + 2)); // More sweeps deeper

    levelParams.set("smoother: params", smootherParams);
}

    RCP<TpetraOperator> mueluPrec =
        MueLu::CreateTpetraPreconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A, *mueluParams);

    RCP<BelosLinearProblem> problem{rcp(new BelosLinearProblem())};

    problem->setOperator(A);
    problem->setLHS(x);
    problem->setRHS(b);
    // problem->setRightPrec(mueluPrec);
    problem->setLeftPrec(mueluPrec);
    problem->setProblem();

    RCP<Teuchos::ParameterList> belosParams = rcp(new Teuchos::ParameterList());
    belosParams->set("Convergence Tolerance", 1e-12);
    belosParams->set("Maximum Iterations", 500);
    belosParams->set("Verbosity", Belos::Errors | Belos::Warnings | Belos::StatusTestDetails);
    belosParams->set("Output Frequency", 1); 
    Belos::SolverFactory<Scalar, TpetraMultiVector, TpetraOperator> factory;
    RCP<BelosSolverManager> solver = factory.create(solverType, belosParams);
    solver->setProblem(problem);

    RCP<Teuchos::Time> solveTimer = Teuchos::TimeMonitor::getNewTimer("Solve Timer");
    Belos::ReturnType ret;
    {
        Teuchos::TimeMonitor timer(*solveTimer);
        ret = solver->solve();
    }
    Teuchos::TimeMonitor::summarize(std::cout);
    if (ret != Belos::Converged)
        std::cerr << "Warning: solver did not converge." << std::endl;
    return x;
}



int main(int argc, char *argv[]) {
    Tpetra::ScopeGuard tpetraScope(&argc, &argv);

    {
        std::cout << "Kokkos execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

        bool generalized = false;
        GlobalOrdinal nx = 10, ny = 10;
        std::string solverType = "GMRES";
        bool test_analytical = false;

        Teuchos::CommandLineProcessor clp(false);
        clp.setOption("nx", &nx, "Number of grid points in x-direction");
        clp.setOption("ny", &ny, "Number of grid points in y-direction");
        clp.setOption("solver", &solverType, "Solver type (CG or GMRES)");
        clp.setOption("test_analytical", "no_test_analytical", &test_analytical, "Test analytical solution (true/false)");
        clp.setOption("generalized", "nongeneralized", &generalized, "Use generalized matrix (true/false)");
        
        if (clp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
            return EXIT_FAILURE;
        }
        
        RCP<const Teuchos::Comm<int>> comm = Teuchos::DefaultComm<int>::getComm();
        const Tpetra::global_size_t numGlobalEntries = static_cast<Tpetra::global_size_t>(nx) * ny;

        const GlobalOrdinal indexBase = 0;

        RCP<const TpetraMapBase> map = rcp(new TpetraMapBase(numGlobalEntries, indexBase, comm));
        RCP<TpetraVector> b;
        RCP<TpetraVector> phi;
        RCP<TpetraVector> n;

        initializeDistributed(map, phi, nx, ny, PhiFunctor{});

        RCP<TpetraCrsMatrix> A;

        if (generalized) {
            if (!test_analytical){
                initializeDistributed(map, n, nx, ny, NFunctor{});
            }else{
                initializeDistributed(map, n, nx, ny, NAnalyticalFunctor{});
            }
            print2File("n", *n, nx, ny);
            initializeDistributed(map, b, nx, ny, RhoFunctor{});
            A = createGeneralizedPoissonMatrix(b->getMap(), n, nx, ny);
        } else {
            initializeDistributed(map, b, nx, ny, RhoConstFunctor{});
            A = createPoissonMatrix(b->getMap(), nx, ny);
        }

        if (!test_analytical) {
            print2File("phi0", *phi, nx, ny);
            A->apply(*phi, *b);
        }
        print2File("rhs", *b, nx, ny);
        auto start = std::chrono::high_resolution_clock::now();

        phi = solve(A, b, solverType); 

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Solve time: " << elapsed.count() << " seconds" << std::endl;

        print2File("phi", *phi, nx, ny);
    }

    return 0;
}
#include <initialization.hpp>
#include <iostream>
#include <matrix.hpp>


#include <Kokkos_Core.hpp>



template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node, typename CalculateFunc>
void initializeDistributed(
    const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> &rowMap,
    Teuchos::RCP<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> &b,
    GlobalOrdinal nx, GlobalOrdinal ny,
    CalculateFunc calculate_func) {


     std::cout << "Running on execution space in initializataoin: " << typeid(ExecutionSpace).name() << std::endl;

    b = Teuchos::rcp(new Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, true));
    const LocalOrdinal numLocalEntries = rowMap->getLocalNumElements();
    auto b_view = b->getLocalViewDevice(Tpetra::Access::OverwriteAll);

    // Copy GIDs to device
    auto gids_host = rowMap->getMyGlobalIndices();
    Kokkos::View<GlobalOrdinal*> gids_dev("gids_dev", gids_host.size());
    Kokkos::deep_copy(gids_dev, Kokkos::View<const GlobalOrdinal*, Kokkos::HostSpace>(gids_host.data(), gids_host.size()));

    Kokkos::parallel_for("InitializeVector", Kokkos::RangePolicy<ExecutionSpace>(0, numLocalEntries),
        KOKKOS_LAMBDA(const LocalOrdinal local_k) {
            const GlobalOrdinal global_k = gids_dev(local_k);
            const GlobalOrdinal i = global_k / ny;
            const GlobalOrdinal j = global_k % ny;

            const double x = static_cast<double>(i) / static_cast<double>(nx - 1);
            const double y = static_cast<double>(j) / static_cast<double>(ny - 1);

            b_view(local_k, 0) = calculate_func(x, y);
        });
}

template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
void print2File(const std::string &label,
                const Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &vec,
                const Teuchos::Comm<int> &comm,
                GlobalOrdinal nx, GlobalOrdinal ny)
{
    using map_type = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
    const auto map = vec.getMap();

    const int myRank = comm.getRank();

    std::ostringstream filename;
    filename << label << "_" << myRank << ".out";
    std::ofstream outFile(filename.str());

    if (!outFile.is_open()) {
        std::cerr << "Rank " << myRank << ": Failed to open file " << filename.str() << std::endl;
        return;
    }
    

    outFile << std::fixed << std::setprecision(12);

    auto localView = vec.getLocalViewHost(Tpetra::Access::ReadOnly);
    auto data1d = Kokkos::subview(localView, Kokkos::ALL(), 0);

    const LocalOrdinal numLocal = map->getLocalNumElements();

    for (LocalOrdinal local_k = 0; local_k < numLocal; ++local_k) {
        GlobalOrdinal global_k = map->getGlobalElement(local_k);

        GlobalOrdinal i = global_k / ny;
        GlobalOrdinal j = global_k % ny;

        Scalar value = data1d(local_k);
        outFile << "(" << i << ", " << j << ", " << value << ")\n";
    }

    outFile.close();
}



RCP<TpetraVector> solve(RCP<TpetraCrsMatrix> A, RCP<TpetraVector> b, std::string solverType = "GMRES")
{
    using Teuchos::ParameterList;
  
    RCP<TpetraVector> x =
        rcp(new TpetraVector(b->getMap(), true));

    RCP<Teuchos::ParameterList> mueluParams = rcp(new Teuchos::ParameterList());
    mueluParams->set("verbosity", "high");
    mueluParams->set("coarse: max size", 32);
    mueluParams->set("cycle type", "W"); 


    int maxLevels = 10;
    mueluParams->set("max levels", maxLevels);

    for (int level = 0; level < maxLevels - 1; ++level) {
        Teuchos::ParameterList& levelParams = mueluParams->sublist("level " + std::to_string(level));
        levelParams.set("smoother: type", "RELAXATION");
        Teuchos::ParameterList smootherParams;
        smootherParams.set("relaxation: type", "Symmetric Gauss-Seidel");
        smootherParams.set("relaxation: damping factor", 1.0);
        smootherParams.set("relaxation: sweeps", level + 2); 
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
    RCP<BelosSolverManager> solver{};

    if (solverType == "CG"){
        solver = rcp(new BelosCGSolver(problem, belosParams));
    }else if (solverType == "GMRES") {
        solver = rcp(new BelosGMRESSolver(problem, belosParams));
    } else {
        throw std::runtime_error("Unsupported solver type: " + solverType);
    }

    RCP<Teuchos::Time> solveTimer = Teuchos::TimeMonitor::getNewTimer("Solve Timer");
    {
    Teuchos::TimeMonitor timer(*solveTimer);
        Belos::ReturnType ret = solver->solve();
    }
    Teuchos::TimeMonitor::summarize(std::cout);
    return x;
}



int main(int argc, char *argv[]) {
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, nullptr);
    Kokkos::initialize(argc, argv);

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
            Kokkos::finalize();
            return EXIT_FAILURE;
        }
        
        RCP<const Teuchos::Comm<int>> comm = Teuchos::DefaultComm<int>::getComm();
        const Tpetra::global_size_t numGlobalEntries = static_cast<Tpetra::global_size_t>(nx) * ny;

        const GlobalOrdinal indexBase = 0;

        RCP<const TpetraMapBase> map = rcp(new TpetraMapBase(numGlobalEntries, indexBase, comm));
        RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap = Tpetra::createUniformContigMap<LocalOrdinal, GlobalOrdinal>(numGlobalEntries, comm); 
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
            print2File("n", *n, *comm, nx, ny);
            initializeDistributed(map, b, nx, ny, RhoFunctor{});
            A = createGeneralizedPoissonMatrix(b->getMap(), b, n, nx, ny);
        } else {
            initializeDistributed(map, b, nx, ny, RhoConstFunctor{});
            A = createPoissonMatrix(b->getMap(), b, nx, ny);
        }

        if (!test_analytical) {
            print2File("phi0", *phi, *comm, nx, ny);
            A->apply(*phi, *b);
        }
        print2File("rhs", *b, *comm, nx, ny);
        auto start = std::chrono::high_resolution_clock::now();

        phi = solve(A, b, solverType); 

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        if (comm->getRank() == 0) {
            std::cout << "Solve time: " << elapsed.count() << " seconds" << std::endl;
        }

        print2File("phi", *(phi->getVectorNonConst(0)), *comm, nx, ny);
    }

    Kokkos::finalize();
    return 0;
}
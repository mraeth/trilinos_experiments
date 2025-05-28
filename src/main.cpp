#include <matrix.hpp>





/**
 * @brief Calculates the spatially varying coefficient n(x,y)
 * for the new test case: n(x,y) = 1 + Sin[2*M_PI*x]^2 + y^2.
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @return The value of n at (x,y).
 */
double calculate_n(double x, double y) {
    return 1.0 + std::pow(std::sin(2.0 * M_PI * x), 2) + y * y;
}

/**
 * @brief Calculates the source term rho(x,y) for the generalized Poisson equation
 * for the new test case.
 * This is derived from the exact solution phi(x,y) = sin(M_PI*x)sin(M_PI*y)
 * and coefficient n(x,y) = 1 + Sin[2*M_PI*x]^2 + y^2.
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @return The value of rho at (x,y).
 */
double calculate_rho(double x, double y) {
    // Term 1: 2 y Cos[M_PI y]
    double term_inside_bracket_part1 = 2.0 * y * std::cos(M_PI * y);

    // Term 2: M_PI (-1 - 2 y^2 + 4 Cos[2 M_PI x] + 3 Cos[4 M_PI x]) Sin[M_PI y]
    double inner_expr_for_term2 = -1.0 - 2.0 * y * y +
                                  4.0 * std::cos(2.0 * M_PI * x) +
                                  3.0 * std::cos(4.0 * M_PI * x);

    double term_inside_bracket_part2 = M_PI * inner_expr_for_term2 * std::sin(M_PI * y);

    // Combine and multiply by M_PI Sin[M_PI x]
    return M_PI * std::sin(M_PI * x) * (term_inside_bracket_part1 + term_inside_bracket_part2);
}


template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
void initializeDistributedRho(
    const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> &rowMap,
    Teuchos::RCP<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> &b,
    GlobalOrdinal nx, GlobalOrdinal ny)
{
    using VectorType = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    b = Teuchos::rcp(new VectorType(rowMap, true)); // true = initialize to zero

    const LocalOrdinal numLocalEntries = rowMap->getLocalNumElements();

    for (LocalOrdinal local_k = 0; local_k < numLocalEntries; ++local_k) {
        GlobalOrdinal global_k = rowMap->getGlobalElement(local_k);
        
        GlobalOrdinal i = global_k / ny;
        GlobalOrdinal j = global_k % ny;

        Scalar value = calculate_rho(
            static_cast<double>(i) / static_cast<double>(nx-1),
            static_cast<double>(j) / static_cast<double>(ny-1));

        b->replaceLocalValue(local_k, value);
    }
}


template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
void initializeDistributedN(
    const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> &rowMap,
    Teuchos::RCP<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> &b,
    GlobalOrdinal nx, GlobalOrdinal ny)
{
    using VectorType = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    b = Teuchos::rcp(new VectorType(rowMap, true)); // true = initialize to zero

    const LocalOrdinal numLocalEntries = rowMap->getLocalNumElements();

    for (LocalOrdinal local_k = 0; local_k < numLocalEntries; ++local_k) {
        GlobalOrdinal global_k = rowMap->getGlobalElement(local_k);
        
        GlobalOrdinal i = global_k / ny;
        GlobalOrdinal j = global_k % ny;

        Scalar value = calculate_n(
            static_cast<double>(i) / static_cast<double>(nx-1),
            static_cast<double>(j) / static_cast<double>(ny-1));
        b->replaceLocalValue(local_k, value);
    }
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

    outFile << std::fixed << std::setprecision(6);

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



RCP<TpetraMultiVector> solve(RCP<TpetraCrsMatrix> A, RCP<TpetraVector> b, int nx, int ny)
{
    using Teuchos::ParameterList;
  
    RCP<TpetraVector> x =
        rcp(new TpetraVector(b->getMap(), true));

    RCP<Teuchos::ParameterList> mueluParams = rcp(new Teuchos::ParameterList());
    mueluParams->set("verbosity", "high");
    mueluParams->set("coarse: max size", 256);
    mueluParams->set("smoother: type", "CHEBYSHEV");
    mueluParams->set("smoother: pre or post", "both");
    mueluParams->set("sa: damping factor", 1.0);
    mueluParams->set("aggregation: type", "uncoupled");

    Teuchos::ParameterList &chebyParams = mueluParams->sublist("smoother: params");
    chebyParams.set("chebyshev: degree", 2);
    chebyParams.set("chebyshev: eigenvalue max iterations", 10);

    RCP<TpetraOperator> mueluPrec =
        MueLu::CreateTpetraPreconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A, *mueluParams);

    RCP<BelosLinearProblem> problem{rcp(new BelosLinearProblem())};

    problem->setOperator(A);
    problem->setLHS(x);
    problem->setRHS(b);
    problem->setRightPrec(mueluPrec);
    problem->setLeftPrec(mueluPrec);
    problem->setProblem();

    RCP<Teuchos::ParameterList> belosParams = rcp(new Teuchos::ParameterList());
    belosParams->set("Convergence Tolerance", 1e-10);
    belosParams->set("Maximum Iterations", 500);
    belosParams->set("Verbosity", Belos::FinalSummary);
    belosParams->set("Output Frequency", 10);
    belosParams->set("Output Style", Belos::Brief);

    RCP<BelosSolverManager> solver =
        rcp(new Belos::PseudoBlockGmresSolMgr<Scalar, Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>, Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(problem, belosParams));
    Belos::ReturnType ret = solver->solve();

    return x;
}



int main(int argc, char *argv[]) {
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, nullptr);
    Kokkos::initialize(argc, argv);

    {
        bool generalized = false; // Initial value
        int nx = 10, ny = 10;
        Teuchos::CommandLineProcessor clp(false);
        clp.setOption("nx", &nx, "Number of grid points in x-direction");
        clp.setOption("ny", &ny, "Number of grid points in y-direction");
        // Add this line to include 'generalized' as a command-line option
        clp.setOption("generalized", "nongeneralized", &generalized, "Use generalized matrix (true/false)");
        if (clp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
            Kokkos::finalize();
            return EXIT_FAILURE;
        }
        // Get MPI communicator
        RCP<const Teuchos::Comm<int>> comm = Teuchos::DefaultComm<int>::getComm();
        const Tpetra::global_size_t numGlobalEntries = static_cast<Tpetra::global_size_t>(nx) * ny;
        const GlobalOrdinal indexBase = 0;

        // Create Tpetra map
        RCP<const TpetraMapBase> map = rcp(new TpetraMapBase(numGlobalEntries, indexBase, comm));
        RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap =
            Tpetra::createUniformContigMap<LocalOrdinal, GlobalOrdinal>(numGlobalEntries, comm);

        Teuchos::RCP<TpetraVector> b;
        Teuchos::RCP<TpetraVector> n;

        initializeDistributedRho<double, LocalOrdinal, GlobalOrdinal, Node>(map, b, nx, ny);

        print2File<Scalar, LocalOrdinal, GlobalOrdinal, Node>("rhs", *b, *comm, nx, ny);

        RCP<TpetraCrsMatrix> A;

        if (generalized){
            initializeDistributedN<double, LocalOrdinal, GlobalOrdinal, Node>(map, n, nx, ny);
            print2File<Scalar, LocalOrdinal, GlobalOrdinal, Node>("n", *n, *comm, nx, ny);

            A = createGeneralizedPoissonMatrix(b->getMap(), b,n, nx, ny);
        } else {
            A = createPoissonMatrix(b->getMap(), b, nx, ny);
        }

        Teuchos::RCP<Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> phi =
            solve(A, b, nx, ny); 


        print2File<Scalar, LocalOrdinal, GlobalOrdinal, Node>("phi", *(phi->getVectorNonConst(0)), *comm, nx, ny);
    }

    Kokkos::finalize();
    return 0;
}

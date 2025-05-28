#include <fstream>
#include <cmath>

#include <Teuchos_RCP.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>

#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include <BelosSolverFactory.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>

#include <MueLu.hpp>
#include <MueLu_Level.hpp>
#include <MueLu_BaseClass.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_Utilities.hpp>
#include <MueLu_Hierarchy.hpp>
#include <MueLu_TpetraOperator.hpp>

#include <Kokkos_Core.hpp>

#include <BelosTpetraAdapter.hpp>
#include <Tpetra_MultiVector.hpp>

#include <MueLu_CreateTpetraPreconditioner.hpp>
using Scalar = Tpetra::Vector<>::scalar_type;
using GlobalOrdinal = Tpetra::Vector<>::global_ordinal_type;
using TpetraMapBase = Tpetra::Map<>;
using Node = Tpetra::Map<>::node_type;
using LocalOrdinal = Tpetra::Map<>::local_ordinal_type;

using TpetraMap = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
using TpetraCrsGraph = Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>;
using TpetraCrsMatrix = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
using TpetraVector = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
using TpetraMultiVector = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
using TpetraOperator = Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
using TpetraImporter = Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;

using BelosLinearProblem = Belos::LinearProblem<Scalar, TpetraMultiVector, TpetraOperator>;
using BelosSolverManager = Belos::SolverManager<Scalar, TpetraMultiVector, TpetraOperator>;
using BelosCGSolver = Belos::PseudoBlockCGSolMgr<Scalar, TpetraMultiVector, TpetraOperator>;

using TeuchosComm = Teuchos::Comm<int>;
using TeuchosParameterList = Teuchos::ParameterList;

using Teuchos::RCP;
using Teuchos::rcp;

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


RCP<TpetraCrsMatrix> createPoissonMatrix(Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap, RCP<TpetraVector> b, int nx, int ny){
    const GlobalOrdinal globalRowMin = rowMap->getMinGlobalIndex();
    const GlobalOrdinal globalRowMax = rowMap->getMaxGlobalIndex();

    const size_t maxNumEntriesPerRow = 5;

    auto A = rcp(new TpetraCrsMatrix(rowMap, maxNumEntriesPerRow));

    for (GlobalOrdinal globalRow = globalRowMin; globalRow <= globalRowMax; ++globalRow)
    {
        const GlobalOrdinal i = globalRow % nx;
        const GlobalOrdinal j = globalRow / nx;

        Teuchos::Array<GlobalOrdinal> colIndices;
        Teuchos::Array<Scalar> values;

        bool isBoundary = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1);

        if (isBoundary)
        {
            colIndices.push_back(globalRow);
            values.push_back(1.0);
            A->insertGlobalValues(globalRow, colIndices(), values());
            b->replaceGlobalValue(globalRow, 0.0);
        }
        else
        {
            colIndices.push_back(globalRow);
            values.push_back(4.0);

            if (i > 0)
            {
                colIndices.push_back(globalRow - 1);
                values.push_back(-1.0);
            }
            if (i < nx - 1)
            {
                colIndices.push_back(globalRow + 1);
                values.push_back(-1.0);
            }
            if (j > 0)
            {
                colIndices.push_back(globalRow - nx);
                values.push_back(-1.0);
            }
            if (j < ny - 1)
            {
                colIndices.push_back(globalRow + nx);
                values.push_back(-1.0);
            }
            A->insertGlobalValues(globalRow, colIndices(), values());
        }
    }

    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set("Optimize Storage", true);
    A->fillComplete(rowMap, rowMap, params);

    return A;
}



RCP<TpetraMultiVector> run(RCP<TpetraVector> b, int nx, int ny)
{
    using Teuchos::ParameterList;

    auto A = createPoissonMatrix(b->getMap(), b, nx, ny);
  
    RCP<TpetraVector> x =
        rcp(new TpetraVector(b->getMap(), true));

    RCP<Teuchos::ParameterList> mueluParams = rcp(new Teuchos::ParameterList());
    mueluParams->set("verbosity", "medium");
    mueluParams->set("coarse: max size", 32);
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
    belosParams->set("Convergence Tolerance", 1e-7);
    belosParams->set("Maximum Iterations", 200);
    belosParams->set("Verbosity", Belos::FinalSummary);
    belosParams->set("Output Frequency", 10);
    belosParams->set("Output Style", Belos::Brief);

    RCP<BelosSolverManager> solver =
        rcp(new Belos::PseudoBlockGmresSolMgr<Scalar, Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>, Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(problem, belosParams));
    Belos::ReturnType ret = solver->solve();

    return x;
}



template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
void initializeDistributedVector(
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

        Scalar value = std::cos(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(nx)) *
                       std::cos(2.0 * M_PI * static_cast<double>(j) / static_cast<double>(ny));

        b->replaceLocalValue(local_k, value);
    }
}


int main(int argc, char *argv[]) {
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, nullptr);
    Kokkos::initialize(argc, argv);

    {
        int nx = 10, ny = 10;
        Teuchos::CommandLineProcessor clp(false);
        clp.setOption("nx", &nx, "Number of grid points in x-direction");
        clp.setOption("ny", &ny, "Number of grid points in y-direction");
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

        initializeDistributedVector<double, LocalOrdinal, GlobalOrdinal, Node>(map, b, nx, ny);

        print2File<Scalar, LocalOrdinal, GlobalOrdinal, Node>("rhs", *b, *comm, nx, ny);
        Teuchos::RCP<Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> phi =
            run(b, nx, ny); // The run function should return a MultiVector
        print2File<Scalar, LocalOrdinal, GlobalOrdinal, Node>("phi", *(phi->getVectorNonConst(0)), *comm, nx, ny);
    }

    Kokkos::finalize();
    return 0;
}

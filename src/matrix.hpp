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


RCP<TpetraCrsMatrix> createPoissonMatrix(Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap, RCP<TpetraVector> b, int nx, int ny);


RCP<TpetraCrsMatrix> createGeneralizedPoissonMatrix(
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap,
    RCP<TpetraVector> b,
    RCP<TpetraVector> n,
    int nx,
    int ny);
    
#pragma once

#include <fstream>

#include <Teuchos_RCP.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>

#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>

#include <MueLu.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>

#include <Kokkos_Core.hpp>

using Scalar        = Tpetra::Vector<>::scalar_type;
using GlobalOrdinal = Tpetra::Vector<>::global_ordinal_type;
using LocalOrdinal  = Tpetra::Map<>::local_ordinal_type;
using Node          = Tpetra::Map<>::node_type;
using TpetraMapBase = Tpetra::Map<>;
using ExecutionSpace = Node::execution_space;

using TpetraMap         = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
using TpetraCrsGraph    = Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>;
using TpetraCrsMatrix   = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
using TpetraVector      = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
using TpetraMultiVector = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
using TpetraOperator    = Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
using TpetraImporter    = Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;

using BelosLinearProblem = Belos::LinearProblem<Scalar, TpetraMultiVector, TpetraOperator>;
using BelosSolverManager = Belos::SolverManager<Scalar, TpetraMultiVector, TpetraOperator>;

using Teuchos::RCP;
using Teuchos::rcp;

RCP<TpetraCrsMatrix> createPoissonMatrix(
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap,
    int nx, int ny);

RCP<TpetraCrsMatrix> createGeneralizedPoissonMatrix(
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap,
    RCP<TpetraVector> n,
    int nx, int ny);

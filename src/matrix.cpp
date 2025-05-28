#include "matrix.hpp"

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


RCP<TpetraCrsMatrix> createGeneralizedPoissonMatrix(
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap,
    RCP<TpetraVector> b,
    RCP<TpetraVector> n,
    int nx,
    int ny)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using TpetraCrsMatrix = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal>;
    using TpetraVector = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal>;

    const GlobalOrdinal globalRowMin = rowMap->getMinGlobalIndex();
    const GlobalOrdinal globalRowMax = rowMap->getMaxGlobalIndex();

    const size_t maxNumEntriesPerRow = 5;

    auto A = rcp(new TpetraCrsMatrix(rowMap, maxNumEntriesPerRow));

    auto n_view = n->getLocalViewHost(Tpetra::Access::ReadOnly);

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
            Scalar n_curr = n_view(rowMap->getLocalElement(globalRow), 0);

            Scalar n_east = 0.0;
            if (i < nx - 1) {
                GlobalOrdinal globalRowEast = globalRow + 1;
                n_east = (n_curr + n_view(rowMap->getLocalElement(globalRowEast), 0)) / 2.0;
            }

            Scalar n_west = 0.0;
            if (i > 0) {
                GlobalOrdinal globalRowWest = globalRow - 1;
                n_west = (n_curr + n_view(rowMap->getLocalElement(globalRowWest), 0)) / 2.0;
            }

            Scalar n_north = 0.0;
            if (j < ny - 1) {
                GlobalOrdinal globalRowNorth = globalRow + nx;
                n_north = (n_curr + n_view(rowMap->getLocalElement(globalRowNorth), 0)) / 2.0;
            }

            Scalar n_south = 0.0;
            if (j > 0) {
                GlobalOrdinal globalRowSouth = globalRow - nx;
                n_south = (n_curr + n_view(rowMap->getLocalElement(globalRowSouth), 0)) / 2.0;
            }

            Scalar diag_coeff = 0.0;

            if (i < nx - 1) {
                colIndices.push_back(globalRow + 1);
                values.push_back(-n_east);
                diag_coeff += n_east;
            }
            if (i > 0) {
                colIndices.push_back(globalRow - 1);
                values.push_back(-n_west);
                diag_coeff += n_west;
            }
            if (j < ny - 1) {
                colIndices.push_back(globalRow + nx);
                values.push_back(-n_north);
                diag_coeff += n_north;
            }
            if (j > 0) {
                colIndices.push_back(globalRow - nx);
                values.push_back(-n_south);
                diag_coeff += n_south;
            }
            colIndices.push_back(globalRow);
            values.push_back(diag_coeff);

            A->insertGlobalValues(globalRow, colIndices(), values());
        }
    }

    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set("Optimize Storage", true);
    A->fillComplete(rowMap, rowMap, params);

    return A;
}

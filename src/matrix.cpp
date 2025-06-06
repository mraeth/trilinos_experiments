#include "matrix.hpp"

RCP<TpetraCrsMatrix> createPoissonMatrix(Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap, RCP<TpetraVector> b, int nx, int ny) {
    
    const GlobalOrdinal globalRowMin = rowMap->getMinGlobalIndex();
    const GlobalOrdinal globalRowMax = rowMap->getMaxGlobalIndex();

    const size_t maxNumEntriesPerRow = 5;

    auto A = Teuchos::rcp(new TpetraCrsMatrix(rowMap, maxNumEntriesPerRow));

    const double hx = 1.0 / static_cast<double>(nx - 1);
    const double hy = 1.0 / static_cast<double>(ny - 1);
    const double h_sq_inv = 1.0 / (hx * hy);

    for (GlobalOrdinal globalRow = globalRowMin; globalRow <= globalRowMax; ++globalRow) {
        const GlobalOrdinal i = globalRow % nx;
        const GlobalOrdinal j = globalRow / nx;

        Teuchos::Array<GlobalOrdinal> colIndices;
        Teuchos::Array<Scalar> values;

        bool isBoundary = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1);

        if (isBoundary) {
            colIndices.push_back(globalRow);
            values.push_back(1.0);
            A->insertGlobalValues(globalRow, colIndices(), values());
        } else {
            colIndices.push_back(globalRow);
            values.push_back(4.0 * h_sq_inv);

            colIndices.push_back(globalRow - 1);
            values.push_back(-1.0 * h_sq_inv);
            colIndices.push_back(globalRow + 1);
            values.push_back(-1.0 * h_sq_inv);
            colIndices.push_back(globalRow - nx);
            values.push_back(-1.0 * h_sq_inv);
            colIndices.push_back(globalRow + nx);
            values.push_back(-1.0 * h_sq_inv);
            
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
    const GlobalOrdinal globalRowMin = rowMap->getMinGlobalIndex();
    const GlobalOrdinal globalRowMax = rowMap->getMaxGlobalIndex();

    const size_t maxNumEntriesPerRow = 5;

    auto A = Teuchos::rcp(new TpetraCrsMatrix(rowMap, maxNumEntriesPerRow));

    auto n_view = n->getLocalViewHost(Tpetra::Access::ReadOnly);

    // Calculate 1/h^2 for a normalized 1x1 domain
    const double hx = 1.0 / static_cast<double>(nx - 1);
    const double hy = 1.0 / static_cast<double>(ny - 1);
    const double h_sq_inv = 1.0 / (hx * hy); // 1/h^2

    for (GlobalOrdinal globalRow = globalRowMin; globalRow <= globalRowMax; ++globalRow) {
        const GlobalOrdinal i = globalRow % nx;
        const GlobalOrdinal j = globalRow / nx;

        Teuchos::Array<GlobalOrdinal> colIndices;
        Teuchos::Array<Scalar> values;

        bool isBoundary = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1);

        if (isBoundary) {
            colIndices.push_back(globalRow);
            values.push_back(1.0); // Boundary conditions usually not scaled by h^2
            A->insertGlobalValues(globalRow, colIndices(), values());
            // If b is your RHS for the solver, its boundary values should be set to 0.0
            // *outside* this function for homogeneous Dirichlet conditions.
        } else {
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

            // Multiply all coefficients by 1/h^2
            if (i < nx - 1) {
                colIndices.push_back(globalRow + 1);
                values.push_back(-n_east * h_sq_inv);
                diag_coeff += n_east;
            }
            if (i > 0) {
                colIndices.push_back(globalRow - 1);
                values.push_back(-n_west * h_sq_inv);
                diag_coeff += n_west;
            }
            if (j < ny - 1) {
                colIndices.push_back(globalRow + nx);
                values.push_back(-n_north * h_sq_inv);
                diag_coeff += n_north;
            }
            if (j > 0) {
                colIndices.push_back(globalRow - nx);
                values.push_back(-n_south * h_sq_inv);
                diag_coeff += n_south;
            }
            
            colIndices.push_back(globalRow);
            values.push_back(diag_coeff * h_sq_inv); // Diagonal also scaled

            A->insertGlobalValues(globalRow, colIndices(), values());
        }
    }

    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set("Optimize Storage", true);
    A->fillComplete(rowMap, rowMap, params);

    return A;
}

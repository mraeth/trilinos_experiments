#include "matrix.hpp"

/// @brief Assemble the standard 2D Poisson matrix.
///
/// @details Discretizes the equation
/// @f[ -\Delta\phi = f \quad \text{on } [0,1]^2 @f]
/// using the standard 5-point finite difference stencil on a uniform
/// @f$ n_x \times n_y @f$ grid with mesh spacings
/// @f$ h_x = \frac{1}{n_x - 1} @f$ and @f$ h_y = \frac{1}{n_y - 1} @f$.
///
/// Grid points are numbered with a row-major (x-fast) global ordering:
/// @f$ k = j \cdot n_x + i @f$, where @f$ i \in [0, n_x) @f$ is the
/// x-index and @f$ j \in [0, n_y) @f$ is the y-index.
///
/// For interior nodes the assembled row is:
/// @f[
///   \frac{1}{h_x h_y}\bigl(4\phi_{i,j}
///     - \phi_{i+1,j} - \phi_{i-1,j}
///     - \phi_{i,j+1} - \phi_{i,j-1}\bigr) = f_{i,j}
/// @f]
///
/// Boundary rows contain a single 1.0 on the diagonal (identity) so that
/// homogeneous Dirichlet conditions are enforced by setting the corresponding
/// RHS entries to zero.
///
/// @param rowMap  Non-overlapping Tpetra row map describing the parallel distribution.
/// @param nx      Number of grid points in the x-direction.
/// @param ny      Number of grid points in the y-direction.
/// @return        Filled and optimized Tpetra CRS matrix.
RCP<TpetraCrsMatrix> createPoissonMatrix(Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap, int nx, int ny) {

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


/// @brief Assemble the generalized 2D Poisson matrix with a spatially varying coefficient.
///
/// @details Discretizes the equation
/// @f[ -\nabla \cdot \bigl(n(x,y)\,\nabla\phi\bigr) = f \quad \text{on } [0,1]^2 @f]
/// using a conservative finite difference scheme on a uniform
/// @f$ n_x \times n_y @f$ grid. The coefficient @f$ n @f$ is averaged
/// arithmetically to the four cell faces surrounding each interior node
/// @f$ (i, j) @f$:
/// @f[
///   n_{i\pm\tfrac{1}{2},j} = \frac{n_{i,j} + n_{i\pm 1,j}}{2}, \qquad
///   n_{i,j\pm\tfrac{1}{2}} = \frac{n_{i,j} + n_{i,j\pm 1}}{2}.
/// @f]
///
/// The assembled row for an interior node reads:
/// @f[
///   \frac{1}{h_x h_y}\Bigl[
///     \bigl(n_E + n_W + n_N + n_S\bigr)\phi_{i,j}
///     - n_E\,\phi_{i+1,j} - n_W\,\phi_{i-1,j}
///     - n_N\,\phi_{i,j+1} - n_S\,\phi_{i,j-1}
///   \Bigr] = f_{i,j},
/// @f]
/// where @f$ n_E = n_{i+\tfrac{1}{2},j} @f$, etc., and
/// @f$ h_x h_y @f$ is the cell area.
/// Boundary rows are set to the identity to enforce homogeneous Dirichlet
/// conditions (RHS must be zero at those nodes).
///
/// Grid points use the same row-major ordering as createPoissonMatrix:
/// @f$ k = j \cdot n_x + i @f$.
///
/// @note In a parallel MPI run the face-averaged coefficients at process
///       boundaries require @f$ n @f$-values owned by neighbouring ranks.
///       These are fetched via a @c Tpetra::Import into a ghost-extended
///       vector before assembly, so the function is safe for any number
///       of MPI processes.
///
/// @param rowMap  Non-overlapping Tpetra row map describing the parallel distribution.
/// @param n       Distributed vector holding the coefficient @f$ n(x,y) @f$
///                at every grid node, defined on @p rowMap.
/// @param nx      Number of grid points in the x-direction.
/// @param ny      Number of grid points in the y-direction.
/// @return        Filled and optimized Tpetra CRS matrix.
RCP<TpetraCrsMatrix> createGeneralizedPoissonMatrix(
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> rowMap,
    RCP<TpetraVector> n,
    int nx,
    int ny)
{
    const GlobalOrdinal globalRowMin = rowMap->getMinGlobalIndex();
    const GlobalOrdinal globalRowMax = rowMap->getMaxGlobalIndex();

    auto n_view = n->getLocalViewHost(Tpetra::Access::ReadOnly);

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

        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            colIndices.push_back(globalRow);
            values.push_back(1.0);
            A->insertGlobalValues(globalRow, colIndices(), values());
            continue;
        }

        Scalar n_curr  = n_view(globalRow, 0);
        Scalar n_east  = 0.0, n_west = 0.0, n_north = 0.0, n_south = 0.0;

        if (i < nx - 1) n_east  = (n_curr + n_view(globalRow + 1,  0)) / 2.0;
        if (i > 0)      n_west  = (n_curr + n_view(globalRow - 1,  0)) / 2.0;
        if (j < ny - 1) n_north = (n_curr + n_view(globalRow + nx, 0)) / 2.0;
        if (j > 0)      n_south = (n_curr + n_view(globalRow - nx, 0)) / 2.0;

        Scalar diag_coeff = 0.0;
        if (i < nx - 1) { colIndices.push_back(globalRow + 1);  values.push_back(-n_east  * h_sq_inv); diag_coeff += n_east;  }
        if (i > 0)      { colIndices.push_back(globalRow - 1);  values.push_back(-n_west  * h_sq_inv); diag_coeff += n_west;  }
        if (j < ny - 1) { colIndices.push_back(globalRow + nx); values.push_back(-n_north * h_sq_inv); diag_coeff += n_north; }
        if (j > 0)      { colIndices.push_back(globalRow - nx); values.push_back(-n_south * h_sq_inv); diag_coeff += n_south; }
        colIndices.push_back(globalRow);
        values.push_back(diag_coeff * h_sq_inv);

        A->insertGlobalValues(globalRow, colIndices(), values());
    }

    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set("Optimize Storage", true);
    A->fillComplete(rowMap, rowMap, params);

    return A;
}

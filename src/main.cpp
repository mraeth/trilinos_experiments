#include <initialization.hpp>
#include <matrix.hpp>

#include <Kokkos_Core.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

template<typename F>
concept GridFunctor = requires(F f, double x, double y) {
    { f(x, y) } -> std::convertible_to<double>;
};

template <GridFunctor CalculateFunc>
Kokkos::View<double**>
initializeView(const std::string& label, int nx, int ny, CalculateFunc calculate_func) {
    Kokkos::View<double**> v(label, nx, ny);
    Kokkos::parallel_for("Initialize", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nx * ny),
        KOKKOS_LAMBDA(int k) {
            const int i = k % nx;
            const int j = k / nx;
            const double x = static_cast<double>(i) / static_cast<double>(nx - 1);
            const double y = static_cast<double>(j) / static_cast<double>(ny - 1);
            v(i, j) = calculate_func(x, y);
        });
    return v;
}

void print2File(const std::string& label, const Kokkos::View<double**>& v) {
    auto v_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);

    std::ofstream outFile(label + ".out");
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file " << label << ".out" << std::endl;
        return;
    }
    outFile << std::fixed << std::setprecision(12);
    for (int j = 0; j < v_host.extent_int(1); ++j)
        for (int i = 0; i < v_host.extent_int(0); ++i)
            outFile << "(" << i << ", " << j << ", " << v_host(i, j) << ")\n";
}

int main(int argc, char *argv[]) {
    PoissonSolver::ScopeGuard scope(argc, argv);

    std::cout << "Kokkos execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    bool generalized     = false;
    bool test_analytical = false;
    int nx = 10, ny = 10;
    std::string solverType = "GMRES";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg.rfind("--nx=", 0) == 0)      nx           = std::stoi(arg.substr(5));
        else if (arg.rfind("--ny=", 0) == 0)      ny           = std::stoi(arg.substr(5));
        else if (arg.rfind("--solver=", 0) == 0)  solverType   = arg.substr(9);
        else if (arg == "--generalized")           generalized  = true;
        else if (arg == "--test_analytical")       test_analytical = true;
    }

    PoissonSolver solver(nx, ny);

    Kokkos::View<double**> rhs("rhs", nx, ny);

    PoissonMatrix A;
    if (generalized) {
        Kokkos::View<double**> n = test_analytical ? initializeView("n", nx, ny, NAnalyticalFunctor{})
                                         : initializeView("n", nx, ny, NFunctor{});
        print2File("n", n);
        rhs = initializeView("rhs", nx, ny, RhoFunctor{});
        A   = solver.buildGeneralizedMatrix(n);
    } else {
        rhs = initializeView("rhs", nx, ny, RhoConstFunctor{});
        A   = solver.buildMatrix();
    }

    if (!test_analytical) {
        Kokkos::View<double**> phi = initializeView("phi", nx, ny, PhiFunctor{});
        print2File("phi0", phi);
        solver.apply(A, phi, rhs);
    }else {
        Kokkos::View<double**> phi = initializeView("phi0", nx, ny, PhiAnalyticalFunctor{});
        print2File("phi0", phi);
    }
    print2File("rhs", rhs);

    Kokkos::View<double**> x("x", nx, ny);
    auto start = std::chrono::high_resolution_clock::now();

    solver.solve(A, rhs, x, solverType);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Solve time: "
              << std::chrono::duration<double>(end - start).count()
              << " seconds" << std::endl;

    print2File("phi", x);

    return 0;
}

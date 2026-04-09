#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_MathematicalConstants.hpp>

// Kokkos 5 provides device-safe wrappers for all <cmath> functions in the
// Kokkos:: namespace and constants in Kokkos::numbers::. Using std:: variants
// inside KOKKOS_INLINE_FUNCTION / KOKKOS_LAMBDA is undefined on GPU devices.

// --- Gaussian Rings ---
KOKKOS_INLINE_FUNCTION
double gaussian_rings_value(double x, double y, int num_rings,
                            double ring_spacing, double sigma,
                            double amplitude, double cx, double cy) {
    double dx = x - cx;
    double dy = y - cy;
    double r = Kokkos::sqrt(dx * dx + dy * dy);

    double value = 0.0;
    for (int i = 1; i <= num_rings; ++i) {
        double ring_radius = i * ring_spacing;
        value += amplitude * Kokkos::exp(-Kokkos::pow((r - ring_radius), 2) / (2.0 * sigma * sigma));
    }

    return value;
}

// --- Gaussian Spiral ---
KOKKOS_INLINE_FUNCTION
double gaussian_spiral_value(double x, double y, double a, double sigma) {
    constexpr double PI = Kokkos::numbers::pi;

    double dx = x - 0.5;
    double dy = y - 0.5;
    double r = Kokkos::sqrt(dx * dx + dy * dy);
    double theta = Kokkos::atan2(dy, dx);

    if (theta < 0) {
        theta += 2 * PI;
    }

    double value = 0.0;

    for (int i = 0; i < 4; ++i) {
        theta += PI / 2.0;

        if (r > 0.05 * (i % 2) && r < (0.5 - 0.03 * ((i + 1) % 2))) {
            double theta_spiral = a * r;
            double dtheta = Kokkos::fmod((theta - theta_spiral + PI), (2 * PI)) - PI;
            if (dtheta < -PI) dtheta += 2 * PI;
            if (dtheta > PI)  dtheta -= 2 * PI;

            double arc_distance = r * dtheta;

            value += Kokkos::exp(-(arc_distance * arc_distance) /
                              (2.0 * sigma * sigma * (r + 0.1) * (r + 0.1)));
        }
    }

    value += Kokkos::exp(-500.0 * Kokkos::pow((r - 0.5), 2.0) / (2.0 * sigma * sigma));

    return value < 1.0 ? value : 1.0;
}

// --- Turbulence Noise ---
KOKKOS_INLINE_FUNCTION
double turbulence_noise(double x, double y, int N, int seed) {
    constexpr double PI = Kokkos::numbers::pi;
    const int M = N / 2 - 1;

    // Hash-based RNG substitute (std::mt19937 is not device-legal)
    auto rand = [=](int i, int j, int k) -> double {
        unsigned int val = i * 73856093 ^ j * 19349663 ^ k * 83492791 ^ seed;
        val = (val >> 13) ^ val;
        val = val * (val * val * 15731 + 789221) + 1376312589;
        return static_cast<double>((val & 0x7fffffff) % 10000) / 10000.0;
    };

    double value = 0.0;

    for (int ix = 1; ix <= M; ++ix) {
        for (int iy = 1; iy <= M; ++iy) {
            double r1 = rand(ix, iy, 1);
            double r2 = rand(ix, iy, 2);
            double r3 = rand(ix, iy, 3);
            double r4 = rand(ix, iy, 4);

            double term_x = Kokkos::sin(2 * PI * ix * (x + r2));
            double term_y = Kokkos::sin(2 * PI * iy * (y + r4));

            double numerator = r1 * term_x * r3 * term_y;
            double denominator = 4 * PI * PI * (ix * ix + iy * iy);

            value += numerator / denominator;
        }
    }

    return value * Kokkos::sin(PI * x) * Kokkos::sin(PI * y)*Kokkos::sin(PI * x) * Kokkos::sin(PI * y);
}

// --- Exponential Density ---
KOKKOS_INLINE_FUNCTION
double exp_dens(double x, double y) {
    constexpr double PI = Kokkos::numbers::pi;
    return Kokkos::exp(2.5 * Kokkos::sin(10 * PI * x) * Kokkos::sin(10 * PI * y));
}

// --- Derived Quantity Functors ---

struct PhiFunctor {
    KOKKOS_INLINE_FUNCTION
    double operator()(double x, double y) const {
        return turbulence_noise(x, y, 128, 1);
    }
};

struct NFunctor {
    KOKKOS_INLINE_FUNCTION
    double operator()(double x, double y) const {
        constexpr double PI = Kokkos::numbers::pi;
        return Kokkos::exp(1000.0 * turbulence_noise(x, y, 32, 5));
    }
};

struct NAnalyticalFunctor {
    KOKKOS_INLINE_FUNCTION
    double operator()(double x, double y) const {
        constexpr double PI = Kokkos::numbers::pi;
        return 0.1 + (Kokkos::sin(2.0 * PI * x) * Kokkos::sin(2.0 * PI * x) +
                      Kokkos::sin(2.0 * PI * y) * Kokkos::sin(2.0 * PI * y));
    }
};

struct RhoFunctor {
    KOKKOS_INLINE_FUNCTION
    double operator()(double x, double y) const {
        constexpr double pi = Kokkos::numbers::pi;
        const double pi_sq = pi * pi;

        return (4.0 * pi_sq * Kokkos::cos(pi * x) * Kokkos::cos(2.0 * pi * x) * Kokkos::sin(2.0 * pi * x) * Kokkos::sin(pi * y)) +
               (4.0 * pi_sq * Kokkos::cos(pi * y) * Kokkos::cos(2.0 * pi * y) * Kokkos::sin(pi * x) * Kokkos::sin(2.0 * pi * y)) -
               (2.0 * pi_sq * Kokkos::sin(pi * x) * Kokkos::sin(pi * y) *
                (0.1 + Kokkos::sin(2.0 * pi * x) * Kokkos::sin(2.0 * pi * x) +
                       Kokkos::sin(2.0 * pi * y) * Kokkos::sin(2.0 * pi * y)));
    }
};

struct RhoConstFunctor {
    KOKKOS_INLINE_FUNCTION
    double operator()(double x, double y) const {
        constexpr double PI = Kokkos::numbers::pi;
        return -2.0 * PI * PI * Kokkos::sin(PI * x) * Kokkos::sin(PI * y);
    }
};


struct PhiAnalyticalFunctor {
    KOKKOS_INLINE_FUNCTION
    double operator()(double x, double y) const {
        constexpr double PI = Kokkos::numbers::pi;
        return -1.0*Kokkos::sin(PI * x) * Kokkos::sin(PI * y);
    }
};
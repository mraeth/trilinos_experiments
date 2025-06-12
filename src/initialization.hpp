#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>

// --- Gaussian Rings ---
KOKKOS_INLINE_FUNCTION
double gaussian_rings_value(double x, double y, int num_rings,
                            double ring_spacing, double sigma,
                            double amplitude, double cx, double cy) {
    double dx = x - cx;
    double dy = y - cy;
    double r = std::sqrt(dx * dx + dy * dy);

    double value = 0.0;
    for (int i = 1; i <= num_rings; ++i) {
        double ring_radius = i * ring_spacing;
        value += amplitude * std::exp(-std::pow((r - ring_radius), 2) / (2.0 * sigma * sigma));
    }

    return value;
}

// --- Gaussian Spiral ---
KOKKOS_INLINE_FUNCTION
double gaussian_spiral_value(double x, double y, double a, double sigma) {
    double dx = x - 0.5;
    double dy = y - 0.5;
    double r = std::sqrt(dx * dx + dy * dy);
    double theta = std::atan2(dy, dx);

    if (theta < 0) {
        theta += 2 * M_PI;
    }

    double value = 0.0;

    for (int i = 0; i < 4; ++i) {
        theta += M_PI / 2.0;

        if (r > 0.05 * (i % 2) && r < (0.5 - 0.03 * ((i + 1) % 2))) {
            double theta_spiral = a * r;
            double dtheta = std::fmod((theta - theta_spiral + M_PI), (2 * M_PI)) - M_PI;
            if (dtheta < -M_PI) dtheta += 2 * M_PI;
            if (dtheta > M_PI) dtheta -= 2 * M_PI;

            double arc_distance = r * dtheta;
            double coeff = (i % 2) ? r : (0.5 - r);

            value += std::exp(-(arc_distance * arc_distance) /
                              (2.0 * sigma * sigma * (r + 0.1) * (r + 0.1)));
        }
    }

    value += std::exp(-500.0 * std::pow((r - 0.5), 2.0) / (2.0 * sigma * sigma));

    return value < 1.0 ? value : 1.0;
}

// --- Turbulence Noise ---
KOKKOS_INLINE_FUNCTION
double turbulence_noise(double x, double y, int N, int seed) {
    const int M = N / 2 - 1;
    const double PI = 3.14159265358979323846;

    // Simple hash-based RNG substitute (since std::mt19937 isn't device-legal)
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

            double term_x = std::sin(2 * PI * ix * (x + r2));
            double term_y = std::sin(2 * PI * iy * (y + r4));

            double numerator = r1 * term_x * r3 * term_y;
            double denominator = 4 * PI * PI * (ix * ix + iy * iy);

            value += numerator / denominator;
        }
    }

    return value * std::sin(M_PI * x) * std::sin(M_PI * y);
}

// --- Exponential Density ---
KOKKOS_INLINE_FUNCTION
double exp_dens(double x, double y) {
    return std::exp(2.5 * std::sin(10 * M_PI * x) * std::sin(10 * M_PI * y));
}

// --- Derived Quantity Functors ---

struct PhiFunctor {
    KOKKOS_INLINE_FUNCTION
    double operator()(double x, double y) const {
        // return 1.0;
        return turbulence_noise(x, y, 32, 1);
        // return -1 * gaussian_spiral_value(x,y, 20.0, 0.07);
    }
};

struct NFunctor {
    KOKKOS_INLINE_FUNCTION
    double operator()(double x, double y) const {
        return std::exp(1000.0 * turbulence_noise(x, y, 32, 5));
    }
};

struct NAnalyticalFunctor {
    KOKKOS_INLINE_FUNCTION
    double operator()(double x, double y) const {
        return 0.1 + (std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * x) +
                      std::sin(2.0 * M_PI * y) * std::sin(2.0 * M_PI * y));
    }
};

struct RhoFunctor {
    KOKKOS_INLINE_FUNCTION
    double operator()(double x, double y) const {
        const double pi = M_PI;
        const double pi_sq = pi * pi;

        return (4.0 * pi_sq * std::cos(pi * x) * std::cos(2.0 * pi * x) * std::sin(2.0 * pi * x) * std::sin(pi * y)) +
               (4.0 * pi_sq * std::cos(pi * y) * std::cos(2.0 * pi * y) * std::sin(pi * x) * std::sin(2.0 * pi * y)) -
               (2.0 * pi_sq * std::sin(pi * x) * std::sin(pi * y) *
                (0.1 + std::sin(2.0 * pi * x) * std::sin(2.0 * pi * x) +
                 std::sin(2.0 * pi * y) * std::sin(2.0 * pi * y)));
    }
};

struct RhoConstFunctor {
    KOKKOS_INLINE_FUNCTION
    double operator()(double x, double y) const {
        return -2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
    }
};

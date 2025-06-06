#include "initialization.hpp"
#include <random>
#include <algorithm>
#include <iostream>

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

    return std::min(value, 1.0);
}

double turbulence_noise(double x, double y, int N, int seed) {
    const int M = N / 2 - 1;
    const double PI = 3.14159265358979323846;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(0.0, 1.0);

    double value = 0.0;

    for (int ix = 1; ix <= M; ++ix) {
        for (int iy = 1; iy <= M; ++iy) {
            double r1 = dist(gen);
            double r2 = dist(gen);
            double r3 = dist(gen);
            double r4 = dist(gen);

            double term_x = std::sin(2 * PI * ix * (x + r2));
            double term_y = std::sin(2 * PI * iy * (y + r4));

            double numerator = r1 * term_x * r3 * term_y;
            double denominator = 4 * PI * PI * (ix * ix + iy * iy);

            value += numerator / denominator;
        }
    }

    return value * std::sin(M_PI * x) * std::sin(M_PI * y);
}

double exp_dens(double x, double y) {
    return std::exp(2.5 * std::sin(10 * M_PI * x) * std::sin(10 * M_PI * y));
}

double calculate_phi(double x, double y) {
    return turbulence_noise(x, y);
    // return -1 * gaussian_spiral_value(x,y);
    // return std::sin((3*M_PI * x)*(5*M_PI * x))*std::sin((5*M_PI * y)*(3*M_PI * y)) * std::sin(M_PI * y)*std::sin(M_PI*x);
}

double calculate_n(double x, double y) {
    return std::exp(1000 * turbulence_noise(x, y, 32, 5));
}

double calculate_n_analytical(double x, double y) {
    return 0.1 + (std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * x) +
                  std::sin(2.0 * M_PI * y) * std::sin(2.0 * M_PI * y));
}

double calculate_rho(double x, double y) {
    const double pi = M_PI;
    const double pi_sq = pi * pi;

    return (4.0 * pi_sq * std::cos(pi * x) * std::cos(2.0 * pi * x) * std::sin(2.0 * pi * x) * std::sin(pi * y)) +
           (4.0 * pi_sq * std::cos(pi * y) * std::cos(2.0 * pi * y) * std::sin(pi * x) * std::sin(2.0 * pi * y)) -
           (2.0 * pi_sq * std::sin(pi * x) * std::sin(pi * y) *
            (0.1 + std::sin(2.0 * pi * x) * std::sin(2.0 * pi * x) +
             std::sin(2.0 * pi * y) * std::sin(2.0 * pi * y)));
}

double calculate_rho_const(double x, double y) {
    return -2 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
}

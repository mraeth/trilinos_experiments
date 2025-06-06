#include <cmath>

double gaussian_rings_value(double x, double y, int num_rings = 5,
                            double ring_spacing = 0.1, double sigma = 0.02,
                            double amplitude = 1.0, double cx = 0.5, double cy = 0.5);

double gaussian_spiral_value(double x, double y, double a = 20.0, double sigma = 0.07);

double turbulence_noise(double x, double y, int N = 32, int seed = 1);

double exp_dens(double x, double y);

double calculate_phi(double x, double y);
double calculate_n(double x, double y);
double calculate_n_analytical(double x, double y);
double calculate_rho(double x, double y);
double calculate_rho_const(double x, double y);
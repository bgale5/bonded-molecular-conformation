#include <iostream>
#include <random>
#include <cmath>
#include <lbfgs.h>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <cstring>

#define MAX_LBFGS_ITER 99

//#define DEBUG

struct Cartesian {
	double x, y;
};

struct Bfgs_molecule {
	double* molecule;
	bool* fixed;

	explicit Bfgs_molecule(int n)
	{
		molecule = new double[n];
		fixed = new bool[n];
		memset(molecule, 0, sizeof(molecule) * n);
		memset(fixed, 0, sizeof(fixed) * n);
	}
	~Bfgs_molecule()
	{
		delete [] molecule;
		delete [] fixed;
	}
};

typedef void (* crossover_ptr)(const double*, const double*, double*, double*);
typedef void (* mutation_ptr)(double*);

// --------------------- Function Prototypes -------------------------- //
double lj_potential(Cartesian* cart, int n);
// -----------------------------------------------------------------------

/**
 * This function is completely random and does not consider knots
 */
void randomise_mol(double* molecule, int n)
{
	std::random_device rd;
	std::mt19937 gen(rd()); // New Mersenne-Twister generator seeded with rd
	std::uniform_real_distribution<double> uniform(-M_PI, M_PI);
	for (int i = 0; i < n; i++)
		molecule[i] = uniform(gen);
	molecule[0] = 0;
}

void print_csv(const double* molecule, double energy, int n_angles)
{
	for (auto i = 0; i<n_angles; i++) {
		std::cout << molecule[i] << ",";
	}
	std::cout << energy << std::endl;
}

// Note: Cart should be n_angles + 1 in length
void angle_to_cart(const double* angles, Cartesian* cart, int n_angles, int n_cart)
{
	if (n_cart != n_angles + 1) {
		std::cerr << "Incorrect sizes" << std::endl;
		exit(-1);
	}
	cart[0] = {0, 0};
	double phi = 0;
	for (int i = 0; i<n_angles; i++) {
		phi += angles[i];
		double x = cart[i].x+cos(phi);
		double y = cart[i].y+sin(phi);
		cart[i+1] = {x, y};
	}
}

/**
 * Calculate the Lennard-Jones potential energy of the given system s
 */
double lj_potential(Cartesian* cart, int n)
{
	double v = 0;
	for (int i = 0; i<n-1; i++) {
		for (int j = i+1; j<n; j++) {
			double delta_x = cart[j].x-cart[i].x;
			double delta_y = cart[j].y-cart[i].y;
			v += 1/pow(pow(delta_x, 2)+pow(delta_y, 2), 6)
					-2/pow(pow(delta_x, 2)+pow(delta_y, 2), 3);
		}
	}
	return v;
}

// Length of grads must be equal to s.length-2 and be able to contain doubles
void lj_gradient(const Cartesian* cart, double* gradients, int n_cart, int n_grad)
{
	if (n_cart != n_grad + 1) {
		std::cerr << "Incorrect array sizes" << std::endl;
		exit(-1);
	}
	for (int m = 0; m<n_grad; m++) {
		double dv = 0;
		for (int i = 0; i<m; i++) {
			for (int j = m+1; j<n_cart; j++) {
				double dx = cart[i].x-cart[j].x;
				double dy = cart[i].y-cart[j].y;
				double r_sqrd = pow(dx, 2)+pow(dy, 2);
				double res = (1.0/pow(r_sqrd, 7)-1.0/pow(r_sqrd, 4))
						*(dx*(cart[m].y-cart[j].y)
								+dy*(cart[j].x-cart[m].x));
				dv += res;
			}
		}
		gradients[m] = 12*dv;
	}
}

// The callback interface to provide objective function and gradient evaluations
// for liblbfgs
lbfgsfloatval_t lbfgs_evaluate(
		void* fixed,
		const lbfgsfloatval_t* x,
		lbfgsfloatval_t* g,
		const int n,
		const lbfgsfloatval_t step
)
{
//	auto s = (Molecule*)molecule;
//	lj_gradient(x, g, n);
//	return lj_potential(*s);
}

int lbfgs_progress(
		void* instance,
		const lbfgsfloatval_t* x,
		const lbfgsfloatval_t* g,
		const lbfgsfloatval_t fx,
		const lbfgsfloatval_t xnorm,
		const lbfgsfloatval_t gnorm,
		const lbfgsfloatval_t step,
		int n,
		int k,
		int ls
)
{
#ifdef DEBUG
	printf("Iteration %d:\n", k);
	printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
	printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
	printf("\n");
#endif
	return k >= MAX_LBFGS_ITER;
}

void single_crossover(const double* p1, const double* p2, double* c1, double* c2, int n)
{
	//int cp = rand() % (int)p1->angles.size();
	int cp = 2;
	std::copy(p1, p1+cp, c1);
	std::copy(p2+cp, p2+n, c1+cp);
	std::copy(p2, p2+cp, c2);
	std::copy(p1+cp, p1+n, c2+cp);
}

void single_mutation(double* s)
{
	return;
}

// Create 2 parents, if either is better than the parent, replace the parent
// with it. Constant pop size.
void ga(
		double** population,
		const crossover_ptr* crossovers,
		const mutation_ptr* mutations,
		int n_crossovers,
		int n_mutations,
		int max_gen
)
{
	return;
}

int main()
{
	srand((unsigned int) time(nullptr));
	double molecule[3] = {0, M_PI/6, -M_PI/2};
	//randomise_mol(molecule, 5);
	Cartesian cart[4];
	angle_to_cart(molecule, cart, 3, 4);
	double energy = lj_potential(cart, 4);
	print_csv(molecule, energy, 3);
	double gradients[3] = {0};
	lj_gradient(cart, gradients, 4, 3);
	for (int i = 0; i < 3; i++)
	{
		std::cout << gradients[i] << std::endl;
	}
	return 0;
}
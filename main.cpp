#include <iostream>
#include <random>
#include <cmath>
#include <lbfgs.h>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <cstring>

#define MAX_LBFGS_ITER 99
#define MISC_BUF 100
#define MUTATION_RATE 0.5f

//#define DEBUG

struct Cartesian {
	double x = 0, y = 0;
};

struct Candidate {
	double *m;
	double f;
};

struct Bfgs_molecule {
	double *molecule;
	double *gradients;
	Cartesian *cart;
	int *indices;
	int n_angles;
	int n_cart;
	int n_indices;
	int n_gradient;
};

typedef void (*crossover_ptr)(
	const double*,
	const double*,
	double*,
	int,
	double*,
	Cartesian*,
	int,
	int
);

typedef void (*mutation_ptr)(double*, int, Cartesian*, int);

// --------------------- Function Prototypes -------------------------- //
double lj_potential(Cartesian *cart, int n);
// -----------------------------------------------------------------------

/**
 * This function is completely random and does not consider knots
 */
void randomise_mol(double *molecule, int n)
{
	std::random_device rd;
	std::mt19937 gen(rd()); // New Mersenne-Twister generator seeded with rd
	std::uniform_real_distribution<double> uniform(-M_PI, M_PI);
	for (int i = 0; i<n; i++)
		molecule[i] = uniform(gen);
	molecule[0] = 0;
}

void print_csv(const double *molecule, double energy, int n_angles)
{
	for (auto i = 0; i<n_angles; i++) {
		std::cout << molecule[i] << ",";
	}
	std::cout << energy << std::endl;
}

// Note: Cart should be n_angles + 1 in length
void angle_to_cart(const double *angles, Cartesian *cart, int n_angles, int n_cart)
{
	if (n_cart!=n_angles+1) {
		std::cerr << "Incorrect sizes" << std::endl;
		exit(-1);
	}
	cart[0].x = 0;
	cart[0].y = 0;
	double phi = 0;
	for (int i = 0; i<n_angles; i++) {
		phi += angles[i];
		double x = cart[i].x+cos(phi);
		double y = cart[i].y+sin(phi);
		cart[i+1].x = x;
		cart[i+1].y = y;
	}
}

/**
 * Calculate the Lennard-Jones potential energy of the given system s
 */
double lj_potential(Cartesian *cart, int n)
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
void lj_gradient(const Cartesian *cart, double *gradients, int n_cart, int n_grad)
{
	if (n_cart!=n_grad+1) {
		std::cerr << "Incorrect array sizes" << std::endl;
		exit(-1);
	}
	for (int m = 0; m<n_grad; m++) {
		double dv = 0;
		for (int i = 0; i < m; i++) {
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
		void *instance,
		const lbfgsfloatval_t *x,
		lbfgsfloatval_t *g,
		const int n,
		const lbfgsfloatval_t step
)
{
	auto bm = (Bfgs_molecule*) instance;
	if (bm->indices!=nullptr && n!=bm->n_indices) {
		std::cerr << "Parameter size doesn't match number of indices"
		          << std::endl;
		exit(-1);
	}
	if (bm->indices!=nullptr) {
		for (int i = 0; i < bm->n_indices; i++) {
			bm->molecule[bm->indices[i]] = x[i];
		}
		angle_to_cart(bm->molecule, bm->cart, bm->n_angles, bm->n_cart);
		lj_gradient(bm->cart, bm->gradients, bm->n_cart, bm->n_gradient);
		for (int i = 0; i < bm->n_indices; i++) {
			g[i] = bm->gradients[bm->indices[i]];
		}
	} else {
		angle_to_cart(x, bm->cart, n, bm->n_cart);
		lj_gradient(bm->cart, g, bm->n_cart, bm->n_gradient);
	}
	return lj_potential(bm->cart, bm->n_cart);
}

int lbfgs_progress(
		void *instance,
		const lbfgsfloatval_t *x,
		const lbfgsfloatval_t *g,
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
	return k>=MAX_LBFGS_ITER;
}

void single_crossover(
		const double *p1,
		const double *p2,
		double *c,
		int n,
		double *grads,
		Cartesian *cart,
		int n_grad,
		int n_cart
)
{
	int cp = rand()%n;
	std::copy(p1, p1+cp, c);
	std::copy(p2+cp, p2+n, c+cp);

	// Optimise at the crossover point
	int indices[1] = {cp};
	Bfgs_molecule bm = {c, grads, cart, indices, n, n+1, 1, n};
	lbfgsfloatval_t res;
	lbfgs(1, c+cp, &res, lbfgs_evaluate, lbfgs_progress, &bm, nullptr);
}

//void single_mutation(double *m, int n, Cartesian *cart, int n_cart)
//{
//	if (n != n_cart - 1) {
//		std::cerr << "Sizes don't match" << std::endl;
//		exit(-1);
//	}
//	std::random_device rd;
//	std::mt19937 gen(rd()); // New Mersenne-Twister generator seeded with rd
//	std::uniform_real_distribution<double> uniform(-M_PI, M_PI);
//	angle_to_cart(m, cart, n, n_cart);
//	double initial_e = lj_potential(cart, n);
//	double e;
//	do {
//		int mp = rand() % n;
//		double prev = m[mp];
//		m[mp] = uniform(gen);
//		angle_to_cart(m, cart, n, n_cart);
//		e = lj_potential(cart, n_cart);
//		if (e > initial_e) // Revert
//			m[mp] = prev;
//	} while (e > initial_e);
//}

void single_mutation(double *m, int n, Cartesian *cart, int n_cart)
{
	if (n != n_cart - 1) {
		std::cerr << "Sizes don't match" << std::endl;
		exit(-1);
	}
	std::random_device rd;
	std::mt19937 gen(rd()); // New Mersenne-Twister generator seeded with rd
	std::uniform_real_distribution<double> uniform(-M_PI/180.0, M_PI/180.0);
	int mp = rand() % n;
	if (mp < n - mp) {
		for (int i = 0; i < mp; i++) {
			m[i] = uniform(gen);
		}
	} else {
		for (int i = mp; i < n; i++) {
			m[i] = uniform(gen);
		}
	}
}



// Create 2 parents, if either is better than the parent, replace the parent
// with it. Constant pop size.
void ga(
		const crossover_ptr *cross,
		const mutation_ptr *mut,
		double *sol,
		int n_pop,
		int n_cross,
		int n_mut,
		int n_angles,
		int max_gen
)
{
	int p = n_pop * 2;
	auto pop = new Candidate[p];
	auto grads = new double[n_angles];
	auto fitness_buf = new double[p];
	auto cart_buf = new Cartesian[n_angles + 1];
	double global_best = std::numeric_limits<double>::max();
	int best_gen = 0;
	Bfgs_molecule bm;
	for (int i = 0; i < p; i++) {
		pop[i].m = new double[n_angles];
		randomise_mol(pop[i].m, n_angles);
	}
	for (int g = 0; g < max_gen; g++) {
		// Cross each of the best half with a random partner, also from the best half
		for (int c = n_pop; c < p; c++) {
			Candidate *p1 = &pop[rand() % n_pop];
			Candidate *p2 = &pop[rand() % n_pop];
			Candidate *child  = &pop[c];
			cross[rand()%n_cross](p1->m, p2->m, child->m, n_angles, grads, cart_buf, n_angles, n_angles+1);
		}

		// Get all the fitnesses for sorting
		for (int i = 0; i < p; i++) {
			angle_to_cart(pop[i].m, cart_buf, n_angles, n_angles+1);
			pop[i].f = lj_potential(cart_buf, n_angles+1);
		}
		std::sort(pop, pop+p, [](const Candidate &a, const Candidate&b) -> bool {
			return a.f < b.f;
		});

		// Mutate all children with a small chance
		for (int i = 1; i < p; i++) {
			if (fabs(pop[i].f - pop[i-1].f) < 1.0 || pop[i].f > 0)
				mut[rand() % n_mut](pop[i].m, n_angles, cart_buf, n_angles+1);
		}

		// Locally Optimise entire population
		for (int i = 0; i < p; i++) {
			bm.molecule = pop[i].m;
			bm.indices = nullptr;
			bm.gradients = grads;
			bm.cart = cart_buf;
			bm.n_cart = n_angles + 1;
			bm.n_angles = n_angles;
			bm.n_indices = 0;
			bm.n_gradient = n_angles;
			double res;
			lbfgs(n_angles, pop[i].m, &res, lbfgs_evaluate, lbfgs_progress, &bm, nullptr);
		}
	}
	// Get all the fitnesses for sorting
	for (int i = 0; i < p; i++) {
		angle_to_cart(pop[i].m, cart_buf, n_angles, n_angles+1);
		pop[i].f = lj_potential(cart_buf, n_angles+1);
	}
	std::sort(pop, pop+p, [](const Candidate &a, const Candidate&b) -> bool {
		return a.f < b.f;
	});


	// Return the best solution
	std::copy(pop[0].m, pop[0].m + n_angles, sol);

	for (int i = 0; i < n_pop; i++) {
		print_csv(pop[i].m, pop[i].f, n_angles);
	}

	// cleanup
	for (int i = 0; i < n_pop; i++)
		delete [] pop[i].m;
	delete [] pop;
	delete [] fitness_buf;
	delete [] cart_buf;
	delete [] grads;
}

int main()
{
	const int n_angles = 61;
	const int n_cart = n_angles + 1;
	const int n_gens = 50;
	srand((unsigned int) time(nullptr));
	crossover_ptr cross[1] = {single_crossover};
	mutation_ptr mut[1] = {single_mutation};
	double sol[n_angles];

	ga(cross, mut, sol, 6, 1, 1, n_angles, n_gens);
	Cartesian cart[n_cart];
	angle_to_cart(sol, cart, n_angles, n_cart);
	double e = lj_potential(cart, n_cart);
	//print_csv(sol, e, n_angles);

}

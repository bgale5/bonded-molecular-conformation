#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <lbfgs.h>
#include <cstdlib>
#include <ctime>
#include <numeric>


//#define DEBUG

struct Molecule {
	int n_atoms;
	double fitness;
	std::vector<double> angles;
	// TODO: Store the cartesian coordinates so that they aren't recomputed.

	explicit Molecule(int n_atoms)
	{
		this->n_atoms = n_atoms;
		angles.resize((unsigned long) n_atoms-1);
		fitness = std::numeric_limits<double>::max();
	}

	explicit Molecule(double* angles, int n_angles)
	{
		n_atoms = n_angles+1;
		this->angles.resize((unsigned long) n_angles);
		std::copy(angles, angles+n_angles, this->angles.begin());
		fitness = std::numeric_limits<double>::max();
	}
};

struct Cartesian {
	double x, y;
};

typedef void (* crossover_ptr)(const Molecule*, const Molecule*, Molecule*, Molecule*);
typedef void (* mutation_ptr)(Molecule*);

// --------------------- Function Prototypes -------------------------- //
double lj_potential(const Molecule& s);
// -----------------------------------------------------------------------

/**
 * This function is completely random and does not consider knots
 */
void randomise_mol(Molecule* s)
{
	std::random_device rd;
	std::mt19937 gen(rd()); // New Mersenne-Twister generator seeded with rd
	std::uniform_real_distribution<double> uniform(-M_PI, M_PI);
	for (auto& i : s->angles)
		i = uniform(gen);
	s->angles[0] = 0;
	s->fitness = std::numeric_limits<double>::max();
}

template<typename Indexible>
void print_csv(const Indexible& angles, const double energy, int n_angles)
{
	for (auto i = 0; i<n_angles; i++) {
		std::cout << angles[i] << ",";
	}
	std::cout << energy << std::endl;
}

// Note: Ensure that cart has at least s.length elements
void angle_to_cart(const Molecule& s, std::vector<Cartesian>& cart)
{
	cart[0] = {0, 0};
	double phi = 0;
	for (int i = 0; i<s.angles.size(); i++) {
		phi += s.angles[i];
		double x = cart[i].x+cos(phi);
		double y = cart[i].y+sin(phi);
		cart[i+1] = {x, y};
	}
}

/**
 * Calculate the Lennard-Jones potential energy of the given system s
 */
double lj_potential(Molecule& s)
{
	std::vector<Cartesian> cart((unsigned long) s.n_atoms);
	angle_to_cart(s, cart);
	double v = 0;
	for (int i = 0; i<s.n_atoms-1; i++) {
		for (int j = i+1; j<s.n_atoms; j++) {
			double delta_x = cart[j].x-cart[i].x;
			double delta_y = cart[j].y-cart[i].y;
			v += 1/pow(pow(delta_x, 2)+pow(delta_y, 2), 6)
					-2/pow(pow(delta_x, 2)+pow(delta_y, 2), 3);
		}
	}
	s.fitness = v;
	return v;
}

// Length of grads must be equal to s.length-2 and be able to contain doubles
template<typename Indexible>
void lj_gradient(const Molecule& s, Indexible gradients)
{
	std::vector<Cartesian> cart((unsigned long) s.n_atoms);
	angle_to_cart(s, cart);
	for (int m = 0; m<s.angles.size(); m++) {
		double dv = 0;
		for (int i = 0; i<m; i++) {
			for (int j = m+1; j<s.n_atoms; j++) {
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
		void* instance,
		const lbfgsfloatval_t* x,
		lbfgsfloatval_t* g,
		const int n,
		const lbfgsfloatval_t step
)
{
	Molecule s((double*) x, n);
	lj_gradient(s, g);
	return lj_potential(s);
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
	return k>=99;
}

void single_crossover(const Molecule* p1, const Molecule* p2, Molecule* c1, Molecule* c2)
{
	//int cp = rand() % (int)p1->angles.size();
	int cp = 2;
	std::copy(p1->angles.begin(), p1->angles.begin()+cp, c1->angles.begin());
	std::copy(p2->angles.begin()+cp, p2->angles.end(), c1->angles.begin()+cp);
	std::copy(p2->angles.begin(), p2->angles.begin()+cp, c2->angles.begin());
	std::copy(p1->angles.begin()+cp, p1->angles.end(), c2->angles.begin()+cp);
}

void single_mutation(Molecule* s)
{
	unsigned long n = s->angles.size();
	double old_fitness = s->fitness;
	std::random_device rd;
	std::mt19937 gen(rd()); // New Mersenne-Twister generator seeded with rd
	std::uniform_real_distribution<double> uniform(-M_PI, M_PI);
	do {
		s->angles[rand()%n] = uniform(gen);
	} while (old_fitness < 0 && s->fitness > 0); // If it makes a knot, try again.
}

// Create 2 parents, if either is better than the parent, replace the parent
// with it. Constant pop size.
Molecule ga(
		std::vector<Molecule*>& population,
		const std::vector<crossover_ptr>& crossovers,
		const std::vector<mutation_ptr>& mutations,
		int max_gen
)
{
	int n_pop = (int)population.size();
	int n_atoms = population[0]->n_atoms;
	auto child1 = new Molecule(n_atoms);
	auto child2 = new Molecule(n_atoms);

	for (int gen = 0; gen < max_gen; gen++)
	{
		for (auto& m : population) {
			print_csv(m->angles, lj_potential(*m), (int)m->angles.size());
		}
		crossovers[0](population[0], population[1], child1, child2);
		print_csv(child1->angles, lj_potential(*child1), (int)child1->angles.size());
		print_csv(child2->angles, lj_potential(*child2), (int)child2->angles.size());
	}
	delete child1;
	delete child2;
	return *population[0];
}

int main()
{
	srand((unsigned int) time(nullptr));
	std::vector<crossover_ptr> crossovers = {single_crossover};
	std::vector<mutation_ptr> mutations = {single_mutation};
	int n_pop = 2;
	int n_atoms = 6;
	int max_generations = 1;
	std::vector<Molecule*> population((unsigned long)n_pop);
	for (auto& m : population) {
		m = new Molecule(n_atoms);
		randomise_mol(m);
	}

	// Run the genetic algorithm
	Molecule fittest = ga(population, crossovers, mutations, max_generations);
	//print_csv(fittest.angles, fittest.fitness, (int)fittest.angles.size());

	// cleanup
	for (auto& m : population) {
		delete m;
	}
	return 0;
}
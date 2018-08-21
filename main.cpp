#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <lbfgs.h>


//#define DEBUG

struct Molecule {
	int n_atoms;
	std::vector<double> bond_angles;

	explicit Molecule(int n_atoms)
	{
		this->n_atoms = n_atoms;
		bond_angles.resize((unsigned long)n_atoms - 1);
	}

	explicit  Molecule(double* angles, int n_angles)
	{
		n_atoms = n_angles + 1;
		bond_angles.resize((unsigned long)n_angles);
		std::copy(angles, angles + n_angles, this->bond_angles.begin());
	}
};

struct Cartesian {
    double x, y;
};

// --------------------- Function Prototypes -------------------------- //
double lj_potential(const Molecule& s);

/**
 * This function is completely random and does not consider knots
 */
Molecule rand_mol(int len)
{
	Molecule s(len);
	std::random_device rd;
	std::mt19937 gen(rd()); // New Mersenne-Twister generator seeded with rd
	std::uniform_real_distribution<double> uniform(-M_PI, M_PI);
	for (auto& i : s.bond_angles)
		i = uniform(gen);
	s.bond_angles[0] = 0;
	return s;
}

template <typename Iterable>
void print_csv(const Iterable& angles, const double energy, int n_angles)
{
	for (auto i = 0; i < n_angles; i++) {
		std::cout << angles[i] << ",";
	}
	std::cout << energy << std::endl;
}


// Note: Ensure that cart has at least s.length elements
void angle_to_cart(const Molecule& s, std::vector<Cartesian> &cart)
{
	cart[0] = {0, 0};
	double phi = 0;
	for (int i = 0; i < s.bond_angles.size(); i++) {
		phi += s.bond_angles[i];
		double x = cart[i].x + cos(phi);
		double y = cart[i].y + sin(phi);
		cart[i + 1] = {x, y};
	}
}

/**
 * Calculate the Lennard-Jones potential energy of the given system s
 */
double lj_potential(const Molecule& s)
{
	std::vector<Cartesian> cart((unsigned long)s.n_atoms);
	angle_to_cart(s, cart);
    double v = 0;
    for (int i = 0; i < s.n_atoms - 1; i++) {
        for (int j = i + 1; j < s.n_atoms; j++) {
        	double delta_x = cart[j].x - cart[i].x;
        	double delta_y = cart[j].y - cart[i].y;
            v += 1 / pow(pow(delta_x, 2) + pow(delta_y, 2), 6)
                 - 2 / pow(pow(delta_x, 2) + pow(delta_y, 2), 3);
        }
    }
    return v;
}

// Length of grads must be equal to s.length-2 and be able to contain doubles
template<typename Iterable>
void lj_gradient(const Molecule& s, Iterable grads)
{
	std::vector<Cartesian> cart((unsigned long)s.n_atoms);
	angle_to_cart(s, cart);
	for (int m = 0; m < s.bond_angles.size(); m++) {
		double dv = 0;
		for (int i = 0; i < m; i++) {
			for (int j = m+1; j < s.n_atoms; j++) {
				double dx = cart[i].x - cart[j].x;
				double dy = cart[i].y - cart[j].y;
				double r_sqrd = pow(dx, 2) + pow(dy, 2);
				double res = (1.0/pow(r_sqrd, 7) - 1.0/pow(r_sqrd, 4))
						* ( dx * (cart[m].y - cart[j].y)
						+ dy * (cart[j].x - cart[m].x) );
				dv += res;
			}
		}
		grads[m] = 12 * dv;
		//std::cout << "grads[" << m << "]: " << grads[m] << std::endl;
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
	Molecule s((double*)x, n);
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
	return k >= 99;
}

int main()
{
//	Molecule m(5);
//	std::vector<double> angles = {0, M_PI/6, M_PI/6, M_PI/6};
//	m.bond_angles = angles;
	for (int i = 0; i < 10; i++) {
		Molecule m = rand_mol(5);
		double optimal_result = 0;
		lbfgsfloatval_t* x = lbfgs_malloc((int) m.bond_angles.size());
		std::copy(m.bond_angles.begin(), m.bond_angles.end(), x);
		lbfgs_parameter_t param;
		lbfgs_parameter_init(&param);
		int status = lbfgs(
				(int) m.bond_angles.size(),
				x,
				&optimal_result,
				lbfgs_evaluate,
				lbfgs_progress,
				nullptr,
				&param
		);
		print_csv(x, optimal_result, (int) m.bond_angles.size());
#ifdef DEBUG
		if (status == LBFGS_ALREADY_MINIMIZED)
			std::cout << "LBFGS_ALREADY_MINIMIZED" << std::endl;
		else if (status == LBFGSERR_ROUNDING_ERROR)
			std::cout << "LBFGSERR_ROUNDING_ERROR" << std::endl;
		else if (status == LBFGSERR_INCORRECT_TMINMAX)
			std::cout << "LBFGSERR_INCORRECT_TMINMAX" << std::endl;
		else
			std::cout << status << std::endl;
#endif
		lbfgs_free(x);
		}
		return 0;
}
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

struct Molecule {
	unsigned length;
	std::vector<double> bond_angles;

	explicit Molecule(unsigned length)
	{
		this->length = length;
		bond_angles.resize(length - 1);
	}
};

struct Cartesian {
    double x, y;
};

/**
 * This function is completely random and does not consider knots
 */
Molecule rand_mol(unsigned len)
{
	Molecule s(len);
	std::random_device rd;
	std::mt19937 gen(rd()); // New Mersenne-Twister generator seeded with rd
	std::uniform_real_distribution<double> uniform(-M_PI, M_PI);
	for (auto& i : s.bond_angles)
		i = uniform(gen);
	return s;
}

void print_csv(Molecule& s)
{
	for (auto i = 0; i < s.bond_angles.size(); i++) {
		std::cout << s.bond_angles[i];
		if (i != s.bond_angles.size() - 1)
			std::cout << ",";
	}
	std::cout << std::endl;
}

/**
 * Calculate the Lennard-Jones potential energy of the given system s
 */
double lj_potential(Molecule& s)
{
    std::vector<Cartesian> cart(s.length);
    cart[0] = {0, 0};
    double phi = 0;
    for (int i = 0; i < s.bond_angles.size(); i++) {
        phi += s.bond_angles[i];
        double x = cart[i].x + cos(phi);
        double y = cart[i].y + sin(phi);
	    cart[i + 1] = {x, y};
    }
    double v = 0;
    for (int i = 0; i < s.length - 1; i++) {
        for (int j = i + 1; j < s.length; j++) {
        	double delta_x = cart[j].x - cart[i].x;
        	double delta_y = cart[j].y - cart[i].y;
            v += 1 / pow(pow(delta_x, 2) + pow(delta_y, 2), 6) - 2 / pow(pow(delta_x, 2) + pow(delta_y, 2), 3);
        }
    }
    return v;
}

int main()
{
	Molecule m(4);
	std::vector<double> angles = {0, M_PI / 4.0, -M_PI / 2.0};
	m.bond_angles = angles;
	print_csv(m);
	std::cout << lj_potential(m) << std::endl;
	return 0;
}
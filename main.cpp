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
		bond_angles.resize(length);
	}
};

/**
 * This function is completely random and does not consider knots
 */
Molecule rand_mol(unsigned len)
{
	Molecule s(len);
	std::random_device rd;
	std::mt19937 gen(rd()); // New Mersenne-Twister generator seeded with rd
	std::uniform_real_distribution<double> dis(-M_PI, M_PI);
	for (auto &i : s.bond_angles)
		i = dis(gen);
	return s;
}

void print_mol(Molecule &s)
{
	for (auto &i : s.bond_angles)
		std::cout << i << ' ';
	std::cout << std::endl;
}

int main()
{
	Molecule m = rand_mol(30);
	print_mol(m);
	return 0;
}
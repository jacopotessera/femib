/*
*	Gauss.h
*/

#ifndef GAUSS_H_INCLUDED_
#define GAUSS_H_INCLUDED_

#include <functional>
#include <vector>

#include "../dmat/dmat.h"

class Gauss
{
	public:
		Gauss();
		~Gauss();
		Gauss(std::string gaussName, int n, std::vector<dvec> nodes, std::vector<double> weights, double volume);
		double integrate(const std::function<double(dvec)> &f);
	//private:
		int n;
		std::string gaussName;
		std::vector<dvec> nodes;
		std::vector<double> weights;
		double volume;
};

#endif


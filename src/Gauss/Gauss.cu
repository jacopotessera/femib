/*
*	Gauss.cu
*/

#include "Gauss.h"

Gauss::Gauss(){}
Gauss::~Gauss(){}

Gauss::Gauss(std::string gaussName, int n, std::vector<dvec> nodes, std::vector<double> weights, double volume)
{
	this->gaussName = gaussName;
	this->n = n; //TODO: ?
	this->nodes = nodes;
	this->weights = weights;
	this->volume = volume; //TODO: ?
}

double Gauss::integrate(const std::function<double(dvec)> &f)
{
	double integral = 0;
	for(int i=0;i<n;++i)
	{
		integral += weights[i]*f(nodes[i]);
	}
	return integral;
}


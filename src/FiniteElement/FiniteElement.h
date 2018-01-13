/*
*	FiniteElement.h
*/

#ifndef FINITEELEMENT_H_INCLUDED_
#define FINITEELEMENT_H_INCLUDED_

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../affine/affine.h"
#include "../dmat/dmat.h"
#include "../dmat/dmat_impl.h"
#include "../utils/Mesh.h"
#include "../tensorAlgebra/tensorAlgebra.h"
#include "../utils/utils.h"

class FiniteElement
{
	public:
	//private:
		FiniteElement();
		~FiniteElement();
		int size;
		int ambientDim;
		std::string finiteElementName;
		std::function<Nodes(Mesh)> buildNodes;
		std::vector<F> baseFunctions;
		std::vector<dvec> stdNodes;
		bool check();
};

std::vector<F> buildFunctions(const Mesh &mesh, const Nodes &nodes, const std::vector<std::vector<int>> &support, const FiniteElement &f);

#endif


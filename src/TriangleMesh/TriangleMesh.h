/*
*	TriangleMesh.h
*/

#ifndef TRIANGLEMESH_H_INCLUDED_
#define TRIANGLEMESH_H_INCLUDED_

#include <vector>
#include <functional>

#include "../../lib/mini-book.h"

#include "../affine/affine.h"
#include "../Cuda/Cuda.h"
#include "../dmat/dmat.h"
#include "../dmat/dmat_impl.h"
#include "../Gauss/Gauss.h"
#include "../utils/Mesh.h"

class TriangleMesh
{
	public:

		TriangleMesh();
		~TriangleMesh();
		
		TriangleMesh(Mesh mesh, Gauss g);

		void setDim();
		void setMeshDim();
		void setAffineTransformation();
		double getMeshRadius();
		
		void loadOnGPU();

		double integrate(const std::function<double(dvec)> &f);
		double integrate(const std::function<double(dvec)> &f, int n);
		std::vector<std::vector<double>> getBox();
	//private:

		Mesh mesh;

		int dim;
		int meshDim;

		std::vector<dmat> B;
		std::vector<dmat> Binv;
		std::vector<double> Bdet;
		std::vector<dvec> b;
		std::vector<std::vector<dvec>> p;
		std::vector<std::vector<double>> d;

		Gauss gauss;

		dvec *devP;
		std::vector<ditrian *> devT;
		ditrian *devTq;
};

std::vector<ditrian> getEdges(const ditrian &t);

#endif


/*
*	FiniteElementSpace.h
*/

#ifndef FINITEELEMENTSPACE_H_INCLUDED_
#define FINITEELEMENTSPACE_H_INCLUDED_

#include <functional>
#include <string>
#include <map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>

#include "../dmat/dmat.h"
#include "../dmat/dmat_impl.h"
#include "../affine/affine.h"
#include "../Gauss/Gauss.h"

#include "../utils/Mesh.h"
#include "../utils/utils.h"
#include "../FiniteElement/FiniteElement.h"

#include "../TriangleMesh/TriangleMesh.h"

class FiniteElementSpace
{
	public:		
		FiniteElementSpace();
		~FiniteElementSpace();
		FiniteElementSpace(TriangleMesh t, FiniteElement f, Gauss g);
		void buildFiniteElementSpace();
		void setElementDim();
		void setSpaceDim();
		virtual void buildEdge();
		int getIndex(int b, int n);
		int getMiniIndex(int b, int n);
		F operator()(const std::vector<double> &v);
		F operator()(const std::vector<double> &v, int n);
		std::vector<std::vector<double>> operator[](const std::vector<double> &v);
		std::vector<std::vector<int>> collisionDetection(const std::vector<std::vector<dvec>> &X);
		std::vector<int> collisionDetection(const std::vector<dvec> &X);
		std::vector<dvec> getValuesInMeshNodes(const std::vector<double> &a);
		std::vector<std::vector<dvec>> getValuesInGaussNodes(const std::vector<double> &a);
		void calc(const std::vector<double> &a);
		F getPreCalc(int n);
		BaseFunction getBaseFunction(int i, int n);
	//private:
		TriangleMesh T;
		Gauss gauss;
		FiniteElement finiteElement;
		int elementDim;
		int theOtherDim;
		int spaceDim;
		int ambientDim;
		Nodes nodes;
		std::vector<std::vector<int>> support;
		std::vector<F> baseFunction;
		[[deprecated]]
		std::vector<F> functions;
		std::vector<std::vector<dvec>> val;
		std::vector<std::vector<dmat>> Dval;
		std::vector<int> edgeFunctions;
		std::vector<int> edge;
		std::vector<int> notEdge;
		Eigen::SparseMatrix<double> C;
		int nBT;

		std::vector<std::map<dvec,xDx>> preCalc;	
		//std::vector<std::map<dvec,dvec>> xMini;
		//std::vector<std::map<dvec,dmat>> dxMini;
		//void buildMini(const std::vector<double> &v);
		//F mini(int n);
};

#endif


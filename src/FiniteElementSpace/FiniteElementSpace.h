/*
*	FiniteElementSpace.h
*/

#ifndef FINITEELEMENTSPACE_H_INCLUDED_
#define FINITEELEMENTSPACE_H_INCLUDED_

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>

#include "../dmat/dmat.h"
#include "../dmat/dmat_impl.h"
#include "../affine/affine.h"
#include "../Gauss/Gauss.h"
#include "../Gauss/GaussService.h"

#include "../TriangleMesh/TriangleMesh.h"
#include "../TriangleMesh/SimplicialMesh.h"
#include "../FiniteElement/FiniteElement.h"
#include "../FiniteElement/FiniteElementService.h"

#include "../utils/Mesh.h"

#include "../utils/utils.h"
#include "../mongodb/struct.h"

template <MeshType meshType>
class FiniteElementSpace
{
	public:		
		FiniteElementSpace();
		~FiniteElementSpace();
		FiniteElementSpace(SimplicialMesh<meshType> t, FiniteElement f, Gauss g);
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

		SimplicialMesh<meshType> T;
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
		Eigen::SparseMatrix<double> gC(const Eigen::SparseMatrix<double> &S);
		Eigen::SparseMatrix<double> gR(const Eigen::SparseMatrix<double> &S);
		Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> E;
		Eigen::SparseMatrix<double> applyEdgeCondition(const Eigen::SparseMatrix<double> &S);

		int nBT;

		std::vector<std::unordered_map<dvec,xDx>> preCalc;
		//std::vector<std::map<dvec,dvec>> xMini;
		//std::vector<std::map<dvec,dmat>> dxMini;
		//void buildMini(const std::vector<double> &v);
		//F mini(int n);
};

template<MeshType meshType>
miniFE finiteElementSpace2miniFE(const FiniteElementSpace<meshType> &finiteElementSpace);

template<MeshType meshType>
FiniteElementSpace<meshType> miniFE2FiniteElementSpace(const miniFE &mini, GaussService &gaussService, FiniteElementService &finiteElementService);

template<MeshType meshTypeA, MeshType meshTypeB>
Eigen::SparseMatrix<double> compress(const Eigen::SparseMatrix<double> &S, const FiniteElementSpace<meshTypeA> &E, const FiniteElementSpace<meshTypeB> &F);

#endif


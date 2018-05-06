/*
*	SimplicialMesh.h
*/

#ifndef SIMPLICIALMESH_H_INCLUDED_
#define SIMPLICIALMESH_H_INCLUDED_

#include <vector>
#include <functional>

#include "../utils/Mesh.h"
#include "../Gauss/Gauss.h"
#include "TriangleMesh.h"

#define MESH_TYPE_TABLE \
X(Triangular) \
X(Parallelogram) \
X(Quadrilateral) \

#define X(a) a,
enum MeshType {
  MESH_TYPE_TABLE
};
#undef X

template<MeshType T>
class SimplicialMesh
{
	public:

		SimplicialMesh();
		~SimplicialMesh();
		
		SimplicialMesh(Mesh mesh, Gauss g);
		//virtual
		void setTriangleMesh();

		double integrate(const std::function<double(dvec)> &f);
		double integrate(const std::function<double(dvec)> &f, int n);

		dvec toMesh0x(const dvec& x, int n);
		dmat toMesh0dx(const dvec& x, int n);
		bool xInN(const dvec& x, int n);
	//private:

		Mesh mesh;
		Mesh mesh0;
		Gauss gauss;
		TriangleMesh triangleMesh;
		std::vector<std::vector<int> > q2t;

};

#endif


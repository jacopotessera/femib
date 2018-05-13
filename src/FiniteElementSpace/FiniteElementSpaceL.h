/*
*	FiniteElementSpaceL
*/

#ifndef FINITEELEMENTSPACEL_H_INCLUDED_
#define FINITEELEMENTSPACEL_H_INCLUDED_

#include "FiniteElementSpace.h"

template<MeshType meshType>
class FiniteElementSpaceL : public FiniteElementSpace<meshType>
{
	public:
		FiniteElementSpaceL() : FiniteElementSpace<meshType>(){};
		FiniteElementSpaceL(SimplicialMesh<meshType> t, FiniteElement f, Gauss g) : FiniteElementSpace<meshType>(t,f,g){};
		FiniteElementSpaceL<meshType>& operator=(const FiniteElementSpace<meshType> &finiteElementSpace);
		void buildEdge();
};

#endif


/*
*	FiniteElementSpaceS.h
*/

#ifndef FINITEELEMENTSPACES_H_INCLUDED_
#define FINITEELEMENTSPACES_H_INCLUDED_

#include "FiniteElementSpace.h"

enum STRUCTURE_THICKNESS {
	THIN,
	THICK
};

template<MeshType meshType>
class FiniteElementSpaceS : public FiniteElementSpace<meshType>
{

	public:
		STRUCTURE_THICKNESS thickness;
		FiniteElementSpaceS() : FiniteElementSpace<meshType>(){};
		FiniteElementSpaceS(SimplicialMesh<meshType> t, FiniteElement f, Gauss g, STRUCTURE_THICKNESS thickness=THIN) : FiniteElementSpace<meshType>(t,f,g){
			this->thickness = thickness;
		};
		FiniteElementSpaceS& operator=(const FiniteElementSpace<meshType> &finiteElementSpace);
		void buildEdge();
};

#endif


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

class FiniteElementSpaceS : public FiniteElementSpace
{

	public:
		STRUCTURE_THICKNESS thickness;
		FiniteElementSpaceS() : FiniteElementSpace(){};
		FiniteElementSpaceS(TriangleMesh t, FiniteElement f, Gauss g, STRUCTURE_THICKNESS thickness=THIN) : FiniteElementSpace(t,f,g){
			this->thickness = thickness;
		};
		FiniteElementSpaceS& operator=(const FiniteElementSpace &finiteElementSpace);
		void buildEdge();
};

#endif


/*
*	FiniteElementService.h
*/

#ifndef FINITEELEMENTSERVICE_H_INCLUDED_
#define FINITEELEMENTSERVICE_H_INCLUDED_

#include <string>
#include <unordered_map>

#include "../utils/Mesh.h"
#include "../utils/Exception.h"
#include "FiniteElement.h"

class FiniteElementService
{
	public:
		FiniteElementService();
		~FiniteElementService();
		FiniteElement getFiniteElement(std::string f);
	private:
		void createFiniteElementList();
		virtual void createCustomFiniteElementList();
		std::unordered_map<std::string,FiniteElement> finiteElementList;
};

#endif


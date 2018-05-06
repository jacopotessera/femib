/*
*	FiniteElementService.cu
*/

#include "FiniteElementService.h"
#include "createFiniteElement.h"

FiniteElementService::FiniteElementService()
{
	createFiniteElementList();
	createCustomFiniteElementList();
}

FiniteElementService::~FiniteElementService(){}

FiniteElement FiniteElementService::getFiniteElement(std::string f)
{
	std::unordered_map<std::string, FiniteElement>::iterator i = finiteElementList.find(f);
	if(i!=finiteElementList.end())
	{
		return finiteElementList[f];
	}
	else
	{
		throw EXCEPTION("FiniteElementService: "+f+" not found!");
	}
}

void FiniteElementService::createFiniteElementList()
{
	//for(auto finiteElementBuilder : finiteElementBuilderList)
	//{
	//	FiniteElement finiteElement = finiteElementBuilder();
	//	finiteElementList[finiteElement.finiteElementName] = finiteElement;
	//}
	finiteElementList["P0_2d2d"] = createFiniteElementP0_2d2d();
	finiteElementList["P0_2d1d"] = createFiniteElementP0_2d1d();
	finiteElementList["P1_2d2d"] = createFiniteElementP1_2d2d();
	finiteElementList["P1_2d1d"] = createFiniteElementP1_2d1d();
	finiteElementList["P1P0_2d1d"] = createFiniteElementP1P0_2d1d();
	finiteElementList["P2_2d2d"] = createFiniteElementP2_2d2d();
	finiteElementList["P1_1d1d"] = createFiniteElementP1_1d1d();
	finiteElementList["P1_1d2d"] = createFiniteElementP1_1d2d();
	finiteElementList["Q1_2d1d"] = createFiniteElementQ1_2d1d();
}

void FiniteElementService::createCustomFiniteElementList(){}


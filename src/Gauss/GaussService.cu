/*
*	GaussService.cu
*/

#include "GaussService.h"
#include "createGauss.h"

GaussService::GaussService()
{
	createGaussList();
	createCustomGaussList();
}

GaussService::~GaussService(){}

Gauss GaussService::getGauss(const std::string &g)
{
	std::unordered_map<std::string, Gauss>::iterator i = gaussList.find(g);
	if(i!=gaussList.end())
	{
		return gaussList[g];
	}
	else
	{
		throw std::invalid_argument("GaussService: "+g+" not found!");
	}
}

void GaussService::createGaussList()
{
	gaussList["gauss1_1d"] = createGauss1_1d();
	gaussList["gauss2_1d"] = createGauss2_1d();
	gaussList["gauss3_1d"] = createGauss3_1d();
	gaussList["gauss4_1d"] = createGauss4_1d();
	gaussList["gauss5_1d"] = createGauss5_1d();
	gaussList["gauss1_2d"] = createGauss1_2d();
	gaussList["gauss2_2d"] = createGauss2_2d();
	gaussList["gauss3_2d"] = createGauss3_2d();
	gaussList["gauss5_2d"] = createGauss5_2d();
}

void GaussService::createCustomGaussList(){}


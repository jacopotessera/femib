/*
*	GaussService.h
*/

#ifndef GAUSSSERVICE_H_INCLUDED_
#define GAUSSSERVICE_H_INCLUDED_

#include <string>
#include <unordered_map>

#include "Gauss.h"

class GaussService
{
	public:
		GaussService();
		~GaussService();
		Gauss getGauss(const std::string &g);
	private:
		void createGaussList();
		virtual void createCustomGaussList();
		std::unordered_map<std::string,Gauss> gaussList;	
};

#endif


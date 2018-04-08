/*
*	Exception.h
*/

#ifndef EXCEPTION_H_INCLUDED_
#define EXCEPTION_H_INCLUDED_

#include <iostream>
#include <stdexcept>
#include <string>

#include "../../lib/Log.h"
#include "../../lib/stacktrace/stacktrace.h"

#define EXCEPTION(x)	femib::Exception(x,__FILE__,__LINE__,__FUNCTION__)

namespace femib
{

class Exception : public std::invalid_argument
{
public:
	Exception(const std::string & msg,const char* file, int line, const char* function) : std::invalid_argument(msg) {
		logx::Logger::getInstance()->log(file,line,function,msg.c_str(),logx::ERROR);
	}
private:
	std::string stacktrace;
};

}

#endif


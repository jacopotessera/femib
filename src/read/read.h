/*
*	read.h
*/

#ifndef READ_H_INCLUDED_
#define READ_H_INCLUDED_

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../dmat/dmat.h"
#include "../utils/Mesh.h"
#include "../../lib/Log.h"

template<class T, class W>
std::vector<T> read(std::string file);

template<class W>
W castToT(std::string s);

template<class T, class W>
void set(T& a, int i, std::string s);

Mesh readMesh(std::string p,std::string t);
Mesh readMesh(std::string p,std::string t, std::string e);

#endif


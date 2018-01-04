/*
*	read.h
*/

#ifndef READ_OLD_H_INCLUDED_
#define READ_OLD_H_INCLUDED_

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../dmat/dmat.h"
#include "../utils/Mesh.h"

template<class T, class W>
std::vector<T> read_old(std::string file);

template<class W>
W castToT_old(std::string s);

template<class T, class W>
void set_old(T& a, int i, std::string s);

Mesh readMesh_old(std::string p,std::string t);
Mesh readMesh_old(std::string p,std::string t, std::string e);

#endif


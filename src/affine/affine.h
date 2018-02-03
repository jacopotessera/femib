/*
*	affine.h
*/

#ifndef AFFINE_H_INCLUDED_
#define AFFINE_H_INCLUDED_

#include "../dmat/dmat.h"
#include "../utils/Mesh.h"
#include "../../lib/Log.h"

dmat affineB(int n, const Mesh &mesh);
dvec affineb(int n, const Mesh &mesh);

#endif


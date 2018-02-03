/*
*	Cuda.h
*/

#ifndef CUDA_H_INCLUDED_
#define CUDA_H_INCLUDED_

#define MAX_BLOCKS 1024
#define STACK_SIZE 12928
#define HEAP_SIZE 20000000

#include <iostream>

#include "../../lib/Log.h"

class Cuda
{
	public:
		Cuda();
		Cuda(int stackSize, int heapSize);
		~Cuda();
		void getSize();
	private:
		void setStackSize(int stackSize);
		void setHeapSize(int heapSize);
		int getStackSize();
		int getHeapSize();
};

#endif


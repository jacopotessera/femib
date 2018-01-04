/*
*	Cuda.cu
*/

#include "Cuda.h"

Cuda::Cuda()
{
	setStackSize(STACK_SIZE);
	setHeapSize(HEAP_SIZE);
}

Cuda::Cuda(int stackSize, int heapSize)
{
	setStackSize(stackSize);
	setHeapSize(heapSize);
}

Cuda::~Cuda(){}

void Cuda::setStackSize(int stackSize)
{
	cudaDeviceSetLimit(cudaLimitStackSize,stackSize);
}

void Cuda::setHeapSize(int heapSize)
{
	cudaDeviceSetLimit(cudaLimitMallocHeapSize,heapSize*sizeof(double));
}

void Cuda::getSize()
{
	size_t size_stack, size_heap;
	cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);
	cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
	std::cout << "Stack size found to be " << (int)size_stack << std::endl;
	std::cout << "Heap size found to be " << (int)size_heap << std::endl;
}


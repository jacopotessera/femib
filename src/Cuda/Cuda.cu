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
	sLOG_INFO("Setting CUDA stack size to " << stackSize << " ...");
	cudaDeviceSetLimit(cudaLimitStackSize,stackSize);

	sLOG_DEBUG("CUDA stack size found to be " << getStackSize());
}

void Cuda::setHeapSize(int heapSize)
{
	sLOG_INFO("Setting CUDA heap size to " << heapSize << " ...");
	cudaDeviceSetLimit(cudaLimitMallocHeapSize,heapSize*sizeof(double));

	sLOG_DEBUG("CUDA heap size found to be " << getHeapSize());
}

int Cuda::getStackSize()
{
	size_t size_stack;
	cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);
	return (int)size_stack;
}

int Cuda::getHeapSize()
{
	size_t size_heap;
	cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
	return (int)size_heap;
}

void Cuda::getSize()
{
	sLOG_INFO("CUDA stack size found to be " << getStackSize());
	sLOG_INFO("CUDA heap size found to be " << getHeapSize());
}


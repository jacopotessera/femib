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
	std::ostringstream ss;
	ss << "Setting CUDA stack size to " << stackSize << " ...";
	LOG_INFO(ss);
	cudaDeviceSetLimit(cudaLimitStackSize,stackSize);

	std::ostringstream tt;
	tt << "CUDA stack size found to be " << getStackSize();
	LOG_DEBUG(tt);
}

void Cuda::setHeapSize(int heapSize)
{
	std::ostringstream ss;
	ss << "Setting CUDA heap size to " << heapSize << " ...";
	LOG_INFO(ss);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize,heapSize*sizeof(double));

	std::ostringstream tt;
	tt << "CUDA heap size found to be " << getHeapSize();
	LOG_DEBUG(tt);
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
	//size_t size_stack, size_heap;
	//cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);
	//cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
	//std::cout << "Stack size found to be " << (int)size_stack << std::endl;
	//std::cout << "Heap size found to be " << (int)size_heap << std::endl;
	std::ostringstream tt;
	tt << "CUDA stack size found to be " << getStackSize();
	LOG_INFO(tt);
	std::ostringstream ss;
	ss << "CUDA heap size found to be " << getHeapSize();
	LOG_INFO(ss);
}


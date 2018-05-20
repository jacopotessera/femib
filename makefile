CC = /opt/cuda/bin/nvcc --expt-relaxed-constexpr
DEBUG = -G -O3 -Xcompiler=-rdynamic #-g -pg
CFLAGS = -std=c++11
EIGENFLAGS = -I/usr/local/include/eigen3/ -Xcompiler -fopenmp -lgomp
CPPUNITFLAGS = -lcppunit
CUDAFLAGS = -L/opt/cuda/lib64 -lcuda -lcudart
MONGOFLAGS = -L/lib -L/usr/local/lib -I/usr/local/include/mongocxx/v_noabi -I/usr/include/libmongoc-1.0 -I/usr/local/include/bsoncxx/v_noabi -I/usr/include/libbson-1.0 -lmongocxx -lbsoncxx -l:libmongocxx.so._noabi -l:libbsoncxx.so._noabi
LIBLOG = lib/logger/build/Log.o -Xcompiler=-rdynamic
DATETIME=`date +"%Y-%m-%d_%H:%M:%S"`
#ln -rs lib/logger/src/Log.h lib/Log.h
#ln -rs lib/logger/build/Log.o lib/Log.o
VPATH = src/Cuda:src/dmat:src/read:src/affine:src/Gauss:src/TriangleMesh:src/FiniteElement:src/FiniteElementSpace:src/utils:src/tensorAlgebra:src/mongodb:src/Simulation
.PHONY : test clean plot doc prepare find todo logger all
default_target: all

logger:
	@make -C lib/logger/ clean test

all: test/build/testCuda test/build/testDmat test/build/testAffine test/build/testSimplicialMesh test/build/testFiniteElement  test/build/testRead  test/build/testGauss test/build/testTriangleMesh test/build/testTensorAlgebra test/build/testMongo test/build/testUtils test/build/testSim
	@echo "All."

build/mongodb_impl.o: mongodb_impl.cpp mongodb_impl.h
	@echo -n "Compiling $@..."
	@gcc -c -o $@ $< -w $(CFLAGS) $(MONGOFLAGS)
	@echo " Done."

build/mongodb.o: mongodb.cpp mongodb.h
	@echo -n "Compiling $@..."
	@gcc -c -o $@ $< -w $(CFLAGS) $(MONGOFLAGS) 
	@echo " Done."

build/%.o: %.cu %.h
	@echo -n "Compiling $@..."	
	@$(CC) -c -rdc=true -o $@ $< -w $(CFLAGS) $(EIGENFLAGS) $(DEBUG)
	@echo " Done."

test/build/testCuda: test/TestCuda.cu build/Cuda.o
	@echo -n "Compiling $@..."
	@$(CC) -o test/build/testCuda $^ $(LIBLOG) $(CFLAGS) $(CPPUNITFLAGS)
	@echo " Done."

test/build/testDmat: test/TestDmat.cu build/dmat.o build/Cuda.o
	@echo -n "Compiling $@..."
	@$(CC) -o test/build/testDmat $^ $(LIBLOG) $(CFLAGS) $(CPPUNITFLAGS)
	@echo " Done."

test/build/testRead: test/TestRead.cu build/dmat.o build/read.o
	@echo -n "Compiling $@..."
	@$(CC) -o test/build/testRead $^ $(LIBLOG) $(CFLAGS) $(CPPUNITFLAGS)
	@echo " Done."

test/build/testAffine: test/TestAffine.cu build/dmat.o build/read.o build/affine.o
	@echo -n "Compiling $@..."
	@$(CC) -o test/build/testAffine $^ $(LIBLOG) $(CFLAGS) $(CPPUNITFLAGS)
	@echo " Done."

test/build/testGauss: test/TestGauss.cu build/dmat.o build/Gauss.o build/GaussService.o
	@echo -n "Compiling $@..."
	@$(CC) -o test/build/testGauss $^ $(LIBLOG) $(CFLAGS) $(CPPUNITFLAGS)
	@echo " Done."

test/build/testTriangleMesh: test/TestTriangleMesh.cu build/dmat.o build/affine.o build/read.o build/Gauss.o build/GaussService.o build/TriangleMesh.o
	@echo -n "Compiling $@..."
	@$(CC) -o test/build/testTriangleMesh $^ $(LIBLOG) $(CFLAGS) $(CPPUNITFLAGS)
	@echo " Done."

test/build/testTensorAlgebra: test/TestTensorAlgebra.cu build/dmat.o build/affine.o build/read.o build/Gauss.o build/GaussService.o build/TriangleMesh.o build/SimplicialMesh.o build/FiniteElementSpace.o build/FiniteElementSpaceV.o build/FiniteElementSpaceQ.o build/FiniteElementSpaceS.o build/FiniteElementSpaceL.o build/tensorAlgebra.o build/FiniteElement.o build/FiniteElementService.o build/utils.o
	@echo -n "Compiling $@..."
	@$(CC) -o test/build/testTensorAlgebra $^ $(LIBLOG) $(CFLAGS) $(EIGENFLAGS) $(CPPUNITFLAGS)
	@echo " Done."

test/build/testFiniteElement: test/TestFiniteElement.cu build/dmat.o build/affine.o build/read.o build/Gauss.o build/TriangleMesh.o build/SimplicialMesh.o build/SimplicialMesh.o build/FiniteElementSpace.o build/tensorAlgebra.o build/GaussService.o build/FiniteElement.o build/FiniteElementService.o build/utils.o build/FiniteElementSpaceV.o build/FiniteElementSpaceQ.o build/FiniteElementSpaceS.o build/FiniteElementSpaceL.o
	@echo -n "Compiling $@..."
	@$(CC) -o test/build/testFiniteElement $^ $(LIBLOG) $(CFLAGS) $(EIGENFLAGS) $(CPPUNITFLAGS)
	@echo " Done."

test/build/testMongo: test/TestMongo.cu build/dmat.o build/read.o build/mongodb_impl.o build/mongodb.o 
	@echo -n "Compiling $@..."
	@$(CC) -o test/build/testMongo $^ $(LIBLOG) $(CFLAGS) $(CPPUNITFLAGS) $(MONGOFLAGS)
	@echo " Done."

test/build/testUtils: test/TestUtils.cu build/affine.o build/Gauss.o build/GaussService.o build/dmat.o build/Cuda.o build/TriangleMesh.o build/SimplicialMesh.o build/FiniteElementSpace.o build/FiniteElementSpaceV.o build/FiniteElementSpaceQ.o build/FiniteElementSpaceS.o build/FiniteElementSpaceL.o  build/tensorAlgebra.o build/read.o build/FiniteElement.o build/FiniteElementService.o build/utils.o
	@echo -n "Compiling $@..."
	@$(CC) -o test/build/testUtils $^ $(LIBLOG) $(CFLAGS) $(CPPUNITFLAGS) $(EIGENFLAGS)
	@echo " Done."

test/build/testSim: test/TestSim.cu build/affine.o build/Gauss.o build/GaussService.o build/dmat.o build/Cuda.o build/TriangleMesh.o build/SimplicialMesh.o build/FiniteElementSpace.o build/FiniteElementSpaceV.o build/FiniteElementSpaceQ.o build/FiniteElementSpaceS.o build/FiniteElementSpaceL.o  build/tensorAlgebra.o build/read.o build/mongodb_impl.o build/mongodb.o build/FiniteElement.o build/FiniteElementService.o build/Simulation.o
	@echo -n "Compiling $@..."
	@$(CC) -o test/build/testSim $^ build/utils.o $(LIBLOG) $(CFLAGS) $(CPPUNITFLAGS) $(EIGENFLAGS) $(MONGOFLAGS)
	@echo " Done."

test/build/testSimplicialMesh: test/TestSimplicialMesh.cu build/dmat.o build/affine.o build/Gauss.o build/GaussService.o build/TriangleMesh.o build/SimplicialMesh.o build/SimplicialMesh.o
	@echo -n "Compiling $@..."
	@$(CC) -o test/build/testSimplicialMesh $^ $(LIBLOG) $(CFLAGS) $(CPPUNITFLAGS)
	@echo " Done."

test: test/build/testCuda test/build/testDmat test/build/testAffine test/build/testSimplicialMesh test/build/testFiniteElement  test/build/testRead  test/build/testGauss test/build/testTriangleMesh test/build/testTensorAlgebra test/build/testMongo test/build/testUtils test/build/testSim
	@ulimit -s unlimited
	@echo "Setting up Cuda..."
	@./test/build/testCuda
	@echo "Testing dmat..."
	@./test/build/testDmat
	@echo "Testing read..."
	@./test/build/testRead
	@echo "Testing Affine..."
	@./test/build/testAffine
	@echo "Testing Gauss..."
	@./test/build/testGauss
	@echo "Testing TriangleMesh..."
	@./test/build/testTriangleMesh
	@echo "Testing tensorAlgebra..."
	@./test/build/testTensorAlgebra
	@echo "Testing mongo..."
	@./test/build/testMongo
	@echo "Testing utils..."
	@./test/build/testUtils
	@echo "Testing SimplicialMesh..."
	@./test/build/testSimplicialMesh
	@echo "Testing FiniteElement..."
	@./test/build/testFiniteElement
	@echo "Testing Simulation..."
	@./test/build/testSim

profile:
	@echo "Profiling..."
	@valgrind --tool=callgrind test/build/testSim
	#@valgrind --tool=helgrind test/build/testSim
	@echo "Done."
	@touch callgrind
	@mv callgrind callgrind.old.$(DATETIME)
	@mv callgrind.out.* callgrind
	@kcachegrind callgrind

clean:
	@echo -n "Cleaning... "
	@rm -rf build/*
	@rm -rf test/build/*
	@rm -rf test/results/*
	@echo "Done."

plot:
	@chmod +x plot/plotSimulation.py
	@./plot/plotSimulation.py $(ID) $(OP)

doc:
	@echo "Great Scott!"

prepare:
	@echo jacopo | sudo -S pacman --noconfirm -S xterm
	@mkdir build test/build test/results

find:
	@echo "Searching for '"${1}"' ..."
	@find src -type f | xargs grep -in "${1}"

todo:
	@cat TODO.md
	@make find 1=TODO


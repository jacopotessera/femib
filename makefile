CC = /opt/cuda/bin/nvcc --expt-relaxed-constexpr
DEBUG = -G -O0 #-g -pg
CFLAGS = -std=c++11
EIGENFLAGS = -I/usr/local/include/eigen3/ -Xcompiler -fopenmp -lgomp
CPPFLAGS = -lcppunit
CUDAFLAGS = -L/opt/cuda/lib64 -lcuda -lcudart
MONGOFLAGS = -L/lib -L/usr/local/lib -I/usr/local/include/mongocxx/v_noabi -I/usr/include/libmongoc-1.0 -I/usr/local/include/bsoncxx/v_noabi -I/usr/include/libbson-1.0 -lmongocxx -lbsoncxx -l:libmongocxx.so._noabi -l:libbsoncxx.so._noabi
DATETIME=`date +"%Y-%m-%d_%H:%M:%S"`

VPATH = src/Cuda:src/dmat:src/read:src/affine:src/Gauss:src/TriangleMesh:src/FiniteElement:src/FiniteElementSpace:src/utils:src/tensorAlgebra:src/mongodb:src/Simulation
.PHONY : test clean plot doc prepare find

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
	@$(CC) -g -o test/build/testCuda $^ $(CFLAGS) $(CPPFLAGS)
	@echo " Done."

test/build/testDmat: test/TestDmat.cu build/dmat.o build/Cuda.o
	@echo -n "Compiling $@..."
	@$(CC) -g -o test/build/testDmat $^ $(CFLAGS) $(CPPFLAGS)
	@echo " Done."

test/build/testRead: test/TestRead.cu build/dmat.o build/read.o
	@echo -n "Compiling $@..."
	@$(CC) -g -o test/build/testRead $^ $(CFLAGS) $(CPPFLAGS)
	@echo " Done."

test/build/testAffine: test/TestAffine.cu build/dmat.o build/read.o build/affine.o
	@echo -n "Compiling $@..."
	@$(CC) -g -o test/build/testAffine $^ $(CFLAGS) $(CPPFLAGS)
	@echo " Done."

test/build/testGauss: test/TestGauss.cu build/dmat.o build/Gauss.o build/GaussService.o
	@echo -n "Compiling $@..."
	@$(CC) -g -o test/build/testGauss $^ $(CFLAGS) $(CPPFLAGS)
	@echo " Done."

test/build/testTriangleMesh: test/TestTriangleMesh.cu build/dmat.o build/affine.o build/read.o build/Gauss.o build/GaussService.o build/TriangleMesh.o
	@echo -n "Compiling $@..."
	@$(CC) -g -o test/build/testTriangleMesh $^ $(CFLAGS) $(CPPFLAGS)
	@echo " Done."

test/build/testTensorAlgebra: test/TestTensorAlgebra.cu build/dmat.o build/affine.o build/read.o build/Gauss.o build/GaussService.o build/TriangleMesh.o build/FiniteElementSpace.o build/FiniteElementSpaceV.o build/FiniteElementSpaceQ.o build/FiniteElementSpaceS.o build/FiniteElementSpaceL.o build/utils.o build/tensorAlgebra.o build/FiniteElement.o build/FiniteElementService.o
	@echo -n "Compiling $@..."
	@$(CC) -g -o test/build/testTensorAlgebra $^ $(CFLAGS) $(EIGENFLAGS) $(CPPFLAGS)
	@echo " Done."

test/build/testFiniteElement: test/TestFiniteElement.cu build/dmat.o build/affine.o build/read.o build/Gauss.o build/TriangleMesh.o build/utils.o build/FiniteElementSpace.o build/FiniteElementSpaceV.o build/FiniteElementSpaceQ.o build/FiniteElementSpaceS.o build/FiniteElementSpaceL.o build/tensorAlgebra.o build/GaussService.o build/FiniteElement.o build/FiniteElementService.o
	@echo -n "Compiling $@..."
	@$(CC) -g -o test/build/testFiniteElement $^ $(CFLAGS) $(EIGENFLAGS) $(CPPFLAGS)
	@echo " Done."

test/build/testMongo: test/TestMongo.cu build/dmat.o build/read.o build/mongodb_impl.o build/mongodb.o 
	@echo -n "Compiling $@..."
	@$(CC) -g -o test/build/testMongo $^ $(CFLAGS) $(CPPFLAGS) $(MONGOFLAGS)
	@echo " Done."

test/build/testSim: test/TestSim.cu build/affine.o build/Gauss.o build/GaussService.o build/dmat.o build/Cuda.o build/TriangleMesh.o build/FiniteElementSpace.o build/utils.o build/FiniteElementSpaceV.o build/FiniteElementSpaceQ.o build/FiniteElementSpaceS.o build/FiniteElementSpaceL.o  build/tensorAlgebra.o build/read.o build/mongodb_impl.o build/mongodb.o build/FiniteElement.o build/FiniteElementService.o build/Simulation.o
	@echo -n "Compiling $@..."
	@$(CC) -g -o test/build/testSim $^ $(CFLAGS) $(CPPFLAGS) $(EIGENFLAGS) $(MONGOFLAGS)
	@echo " Done."

test: test/build/testCuda test/build/testDmat test/build/testRead test/build/testAffine test/build/testGauss test/build/testTriangleMesh test/build/testFiniteElement test/build/testTensorAlgebra test/build/testMongo test/build/testSim
	@ulimit -s unlimited	
	@echo "Setting up Cuda..."
	@./test/build/testCuda
	@echo "Test dmat..."
	@./test/build/testDmat	
	@echo "Test read..."
	@./test/build/testRead
	@echo "Test Affine..."
	@./test/build/testAffine
	@echo "Test Gauss..."
	@./test/build/testGauss
	@echo "Test TriangleMesh..."
	@./test/build/testTriangleMesh
	@echo "Test FiniteElement..."
	@./test/build/testFiniteElement
	@echo "Test tensorAlgebra..."
	@./test/build/testTensorAlgebra
	@echo "Test mongo..."
	@./test/build/testMongo
	@echo "Test Simulation..."
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
	@chmod +x plot/plot.py
	@./plot/plot.py $(SAVE)

doc:
	@echo "Great Scott!"

prepare:
	@echo jacopo | sudo -S pacman --noconfirm -S xterm
	@mkdir build test/build test/results

find:
	@echo "Searching for '"${1}"' ..."
	@find src -type f | xargs grep -i "${1}"


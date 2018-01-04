/*
*	TestDmat.cu
*/

#include "../lib/cppunit.h"
#include "../lib/mini-book.h"
#include "../src/Cuda/Cuda.h"
#include "../src/dmat/dmat_impl.h"

class TestDmat : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestDmat);
	CPPUNIT_TEST(testPINV);	
	CPPUNIT_TEST(test1D);	
	CPPUNIT_TEST(test2D);
	CPPUNIT_TEST(test3D);
	CPPUNIT_TEST_SUITE_END();

protected:
	void testPINV(void);
	void test1D(void);
	void test2D(void);
	void test3D(void);

public:
	void setUp(void);
	void tearDown(void);

private:

	Cuda c;

	dmat A0;
	dmat A1;
	dmat A2;
	dmat A3;
	dmat A4;

	ditrian mesh[1];
	dvec triangolo[4];
	dvec punto[6];
	ditrian *devT;
	dvec *devP;
	dvec *devX;
	bool N[6];
	bool NN[6];
	bool *devN;
};

void
TestDmat::testPINV(void)
{
	CPPUNIT_ASSERT((A0*pinv(A0))(0,0)==1);
	CPPUNIT_ASSERT((pinv(A1)*A1)(0,0)==1);
	CPPUNIT_ASSERT((A2*pinv(A2))(0,0)==1);
	CPPUNIT_ASSERT(pinv(A3)==inv(A3));
	CPPUNIT_ASSERT(pinv(A4)==inv(A4));
}

void
TestDmat::test1D(void)
{
	CPPUNIT_ASSERT(true);
	CPPUNIT_ASSERT(true);
}

void
TestDmat::test2D(void)
{
	CPPUNIT_ASSERT(true);
	CPPUNIT_ASSERT(true);
}

void
TestDmat::test3D(void)
{
	parallel_accurate<<<6,1>>>(devP,devT,devX,devN);
	HANDLE_ERROR(cudaMemcpy(N,devN,6*sizeof(bool),cudaMemcpyDeviceToHost));

	NN[0] = serial_accurate(triangolo,mesh[0],punto[0]);
	NN[1] = serial_accurate(triangolo,mesh[0],punto[1]);
	NN[2] = serial_accurate(triangolo,mesh[0],punto[2]);	
	NN[3] = serial_accurate(triangolo,mesh[0],punto[3]);
	NN[4] = serial_accurate(triangolo,mesh[0],punto[4]);
	NN[5] = serial_accurate(triangolo,mesh[0],punto[5]);
	CPPUNIT_ASSERT(N[0]==0);
	CPPUNIT_ASSERT(N[1]==1);
	CPPUNIT_ASSERT(N[2]==0);
	CPPUNIT_ASSERT(N[3]==0);
	CPPUNIT_ASSERT(N[4]==1);
	CPPUNIT_ASSERT(N[5]==0);
}

void TestDmat::setUp(void)
{
	A0 = {{5}};
	A1 = {{2},{1}};
	A2 = {{1,2}};
	A3 = {{1,2},{3,1}};
	A4 = {{1.2345,34.232,23.4545},{4.4636,0.4354,11.2323},{12.235,3.2354,2}};

	mesh[0] = {0,1,2,3};

	triangolo[0] = {0,0,0};
	triangolo[1] = {1,0,0};
	triangolo[2] = {0,1,0};
	triangolo[3] = {0,0,1};

	punto[0] = {0.5,0.5,0.5};
	punto[1] = {0,0,1};
	punto[2] = {0.333333333334,0.333333333334,0.333333333334};
	punto[3] = {0.3333333333334,0.3333333333334,0.3333333333334};
	punto[4] = {0.25,0.25,0.25};
	punto[5] = {0.33333334,0.33333334,0.33333334};

	dtrian s;
	s = {triangolo[0],triangolo[1],triangolo[2],triangolo[3]};

	HANDLE_ERROR(cudaMalloc((void**)&devT,sizeof(ditrian)));
	HANDLE_ERROR(cudaMemcpy(devT,mesh,sizeof(ditrian),cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&devP,4*sizeof(dvec)));
	HANDLE_ERROR(cudaMemcpy(devP,triangolo,4*sizeof(dvec),cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&devX,6*sizeof(dvec)));
	HANDLE_ERROR(cudaMemcpy(devX,punto,6*sizeof(dvec),cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&devN,6*sizeof(bool)));
}

void TestDmat::tearDown(void)
{
	//delete ;
}

CPPUNIT_TEST_SUITE_REGISTRATION( TestDmat );

int main(int argc, char* argv[])
{
	// informs test-listener about testresults
	CPPUNIT_NS::TestResult testresult;

	// register listener for collecting the test-results
	CPPUNIT_NS::TestResultCollector collectedresults;
	testresult.addListener (&collectedresults);

	// register listener for per-test progress output
	CPPUNIT_NS::BriefTestProgressListener progress;
	testresult.addListener (&progress);

	// insert test-suite at test-runner by registry
	CPPUNIT_NS::TestRunner testrunner;
	testrunner.addTest (CPPUNIT_NS::TestFactoryRegistry::getRegistry().makeTest ());
	testrunner.run(testresult);

	// output results in compiler-format
	CPPUNIT_NS::CompilerOutputter compileroutputter(&collectedresults, std::cerr);
	compileroutputter.write ();

	// Output XML for Jenkins CPPunit plugin
	std::ofstream xmlFileOut("test/results/cppTestDmatResults.xml");
	XmlOutputter xmlOut(&collectedresults, xmlFileOut);
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}

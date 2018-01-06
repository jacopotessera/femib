/*
*	TestSimulation.cu
*/

#include "../lib/cppunit.h"

#include "../src/Simulation/Simulation.h"
#include "../src/read/read.h"

class TestSimulation : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestSimulation);
	CPPUNIT_TEST(test1);	
	CPPUNIT_TEST_SUITE_END();

protected:
	void test1(void);

public:
	void setUp(void);
	void tearDown(void);

private:
	std::string pV, tV, eV;
	std::string pS, tS, eS;
	Mesh mV, mS;
	Gauss gV, gS;
	GaussService gaussService;
	TriangleMesh triMeshV, triMeshS;
	FiniteElement f_1d;
	FiniteElement f_2d;
	FiniteElementService finiteElementService;
	FiniteElementSpaceV V;
	FiniteElementSpaceQ Q;
	FiniteElementSpaceS S;
	FiniteElementSpaceL L;
	timestep t0;
	timestep t1;
	Simulation s;
	Simulation ss;
	dbconfig db;
	std::string id;

	FiniteElement finElemV;
	FiniteElement finElemQ;
	FiniteElement finElemS;

};

void
TestSimulation::test1(void)
{
	CPPUNIT_ASSERT(true);
	//CPPUNIT_ASSERT(ss.timesteps[2].u[0] == s.timesteps[2].u[0]);
	//CPPUNIT_ASSERT(ss.timesteps[1].x[0] == s.timesteps[1].x[0]);
	//CPPUNIT_ASSERT(ss.timesteps[2].x[0] == s.timesteps[2].x[0]);
}

void TestSimulation::setUp(void)
{

	std::cout << std::endl;
//	pV = "mesh/pVd_unif.mat";
//	tV = "mesh/tVd_unif.mat";
//	eV = "mesh/eVd_unif.mat";

// "funziona" con p3 e dt = 0.001 k = 100 in circa 1500 step
	std::cout << Eigen::nbThreads() << std::endl;
	pV = "mesh/perugiamesh/p3.mat";
	tV = "mesh/perugiamesh/t3.mat";
	eV = "mesh/perugiamesh/e3.mat";

	pS = "mesh/pSd_unif.mat";
	tS = "mesh/tSd_unif.mat";
	eS = "mesh/eSd_unif.mat";

	gV = gaussService.getGauss("gauss5_2d");
	gS = gaussService.getGauss("gauss5_1d");
	mV = readMesh(pV,tV,eV);
	mS = readMesh(pS,tS,eS);

	triMeshV = TriangleMesh(mV,gV);
	triMeshV.loadOnGPU();

	triMeshS = TriangleMesh(mS,gS);
	triMeshS.loadOnGPU();

	finElemQ = finiteElementService.getFiniteElement("P1P0_2d1d");
	finElemV = finiteElementService.getFiniteElement("P2_2d2d");
	finElemS = finiteElementService.getFiniteElement("P1_1d2d");

	V = FiniteElementSpaceV(triMeshV,finElemV,gV);
	V.buildFiniteElementSpace();
	V.buildEdge();
	Q = FiniteElementSpaceQ(triMeshV,finElemQ,gV);
	Q.buildFiniteElementSpace();
	Q.buildEdge();
	S = FiniteElementSpaceS(triMeshS,finElemS,gS);
	S.buildFiniteElementSpace();
	S.buildEdge();
	/*L = FiniteElementSpaceL(triMesh,f_2d,g);
	L.buildFiniteElementSpace();
	L.buildEdge();*/

	Parameters parameters;
	parameters.rho = 1.0;
	parameters.eta = 0.01;
	parameters.deltarho = 0.0;
	parameters.kappa = 10.0;
	parameters.deltat = 0.0001;
	parameters.TMAX = 10000;
	db = {"testSimulation"};
	drop(db);
	id = "1";
	t0.time = 0; t0.id = id;
	t1.time = 1; t1.id = id;

	double gamma = 1.1;
	double R = 0.6;
	double xC = 0.0;
	double yC = 0.0;

	std::vector<dvec> A = read<dvec,double>(pS);
	std::vector<double> x(S.spaceDim);
	/*for(int i=0;i<S.spaceDim/2;++i){
		x[i]=A[i](0)*0.8;
		x[i+S.spaceDim/2]=A[i](1)*0.4;
	}*/
	int AA = A.size() - 1;
	for(int i=0;i<S.spaceDim/2;++i)
	{
		x[i]=gamma*R*cos(A[i](0)/AA*2*M_PI)+xC;
		x[i+S.spaceDim/2]=1/gamma*R*sin(A[i](0)/AA*2*M_PI)+yC;
	}
	t0.x = x;
	t1.x = x;
	t0.u = std::vector<double>(V.spaceDim);
	t1.u = std::vector<double>(V.spaceDim);
	for(int i=0;i<V.spaceDim;i++){
		t0.u[i] = 0;
		t1.u[i] = 0;
	}

	std::cout << "V mesh radius " << V.T.getMeshRadius() << std::endl;
	std::cout << "S mesh radius " << S.T.getMeshRadius() << std::endl;

	s = Simulation(id,db,parameters,V,Q,S,t0,t1);
	//s = Simulation(id,db,parameters,V,Q,S,L,t0,t1);
	//s.getSimulation(db,"");
	s.prepare();
	s.advance(10000);
	/*auto t0 = std::chrono::high_resolution_clock::now();
	s.buildFluidMatricesParallel();
	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "Multithread: "<<std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << std::endl;
	s.buildFluidMatrices();
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "Single thread: "<<std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;*/
}

void TestSimulation::tearDown(void)
{
	//delete ;
}

CPPUNIT_TEST_SUITE_REGISTRATION( TestSimulation );

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
	std::ofstream xmlFileOut("test/results/cppTestSimulationResults.xml");
	XmlOutputter xmlOut(&collectedresults, xmlFileOut);
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}

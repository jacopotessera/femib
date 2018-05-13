/*
*	TestSimulation.cu
*/

#include "../lib/cppunit.h"
#include "../lib/Log.h"

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
	SimplicialMesh<Triangular> triMeshV;
	SimplicialMesh<Triangular> triMeshS;
	FiniteElement f_1d;
	FiniteElement f_2d;
	FiniteElementService finiteElementService;
	FiniteElementSpaceV<Triangular> V;
	FiniteElementSpaceQ<Triangular> Q;
	FiniteElementSpaceS<Triangular> S;
	FiniteElementSpaceL<Triangular> L;
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
	logx::Logger::getInstance()->setLogLevel("src/TriangleMesh/SimplicialMesh.cu",LOG_LEVEL_INFO);
	logx::Logger::getInstance()->setLogLevel("test/TestSim.cu",LOG_LEVEL_INFO);
	logx::Logger::getInstance()->setLogLevel("src/Simulation/Simulation.cu",LOG_LEVEL_DEBUG);
	logx::Logger::getInstance()->setLogLevel("src/FiniteElementSpace/FiniteElementSpaceS.cu",LOG_LEVEL_INFO);
	logx::Logger::getInstance()->setLogLevel("src/FiniteElementSpace/FiniteElementSpaceL.cu",LOG_LEVEL_INFO);
//	pV = "mesh/pVd_unif.mat";
//	tV = "mesh/tVd_unif.mat";
//	eV = "mesh/eVd_unif.mat";

// "funziona" con p3 e dt = 0.001 k = 100 in circa 1500 step
	std::ostringstream ss_omp;	
	ss_omp << "OpenMP threads set to " << Eigen::nbThreads() << ".";
	LOG_DEBUG(ss_omp);


	pV = "mesh/perugia/p3.mat";
	tV = "mesh/perugia/t3.mat";
	eV = "mesh/perugia/e3.mat";

	STRUCTURE_THICKNESS s_thickness = THICK;

	if(s_thickness == THICK){
		pS = "mesh/pS_32_3.mat";
		tS = "mesh/tS_32_3.mat";
		eS = "mesh/eS_32_3.mat";
		gS = gaussService.getGauss("gauss5_2d");
		finElemS = finiteElementService.getFiniteElement("P1_2d2d");
	}
	else if(s_thickness == THIN){
		pS = "mesh/pS_64.mat";
		tS = "mesh/tS_64.mat";
		eS = "mesh/eS_64.mat";
		gS = gaussService.getGauss("gauss5_1d");
		finElemS = finiteElementService.getFiniteElement("P1_1d2d");
	}

	gV = gaussService.getGauss("gauss5_2d");
	mV = readMesh(pV,tV,eV);
	mS = readMesh(pS,tS,eS);

	triMeshV = SimplicialMesh<Triangular>(mV,gV);
	triMeshV.triangleMesh.loadOnGPU();

	triMeshS = SimplicialMesh<Triangular>(mS,gS);
	triMeshS.triangleMesh.loadOnGPU();

	finElemQ = finiteElementService.getFiniteElement("P1P0_2d1d");
	finElemV = finiteElementService.getFiniteElement("P2_2d2d");

	if(finElemS.check())
		LOG_OK("finElemS ok");
	else
		throw EXCEPTION("finElemS!");

	V = FiniteElementSpaceV<Triangular>(triMeshV,finElemV,gV);
	V.buildFiniteElementSpace();
	V.buildEdge();
	Q = FiniteElementSpaceQ<Triangular>(triMeshV,finElemQ,gV);
	Q.buildFiniteElementSpace();
	Q.buildEdge();
	S = FiniteElementSpaceS<Triangular>(triMeshS,finElemS,gS,s_thickness);
	S.buildFiniteElementSpace();
	S.buildEdge();
	//L = FiniteElementSpaceL<Triangular>(triMeshS,finElemS,gS);
	//L.buildFiniteElementSpace();
	//L.buildEdge();

	Parameters parameters;
	parameters.rho = 1.0;
	parameters.eta = 0.01;
	parameters.deltarho = 10.0;
	parameters.kappa = 1.0;
	parameters.deltat = 0.0001;
	parameters.TMAX = 10000;
	db = {"testSimulation"};
	//drop(db);
	id = "";
	t0.time = 0; t0.id = id;
	t1.time = 1; t1.id = id;

	double sigma = 0.75;
	double gamma = 1.1;
	double R = 0.6;
	double xC = 0.0;
	double yC = 0.0;

	std::vector<dvec> A = read<dvec,double>(pS);
	std::vector<double> x(S.spaceDim);

	int AA = A.size() - 1;
	for(int i=0;i<S.spaceDim/2;++i)
	{
		if(s_thickness == THIN){
			x[i]=sigma*gamma*R*cos(A[i](0)/AA*2*M_PI)+xC;
			x[i+S.spaceDim/2]=sigma*(1/gamma)*R*sin(A[i](0)/AA*2*M_PI)+yC;
			std::ostringstream ss;
			ss << i << " :\t[ " << x[i] << " , " << x[i+S.spaceDim/2] << " ]";
			LOG_TRACE(ss);
		}
		else if(s_thickness == THICK){
			x[i]=A[i](0)*gamma*sigma;
			x[i+S.spaceDim/2]=A[i](1)*(1.0/gamma)*sigma;
			std::ostringstream ss;
			ss << i << " :\t[ " << x[i] << " , " << x[i+S.spaceDim/2] << " ]";
			LOG_TRACE(ss);
		}
	}

	t0.x = x;
	t1.x = x;
	t0.u = std::vector<double>(V.spaceDim);
	t1.u = std::vector<double>(V.spaceDim);
	for(int i=0;i<V.spaceDim;i++){
		t0.u[i] = 0;
		t1.u[i] = 0;
	}

	std::ostringstream ss,tt;
	ss << "V mesh radius " << V.T.triangleMesh.getMeshRadius();
	tt << "S mesh radius " << S.T.triangleMesh.getMeshRadius();
	LOG_INFO(ss);
	LOG_INFO(tt);

	s = Simulation(db,parameters,V,Q,S,t0,t1);
	//s = Simulation(db,parameters,V,Q,S,L,t0,t1);
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
	std::cout << "Single thread: "<<std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
	double tt1 = (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	double tt2 = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	std::cout << "Ratio: " << (double)(tt1/tt2) << std::endl;*/
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


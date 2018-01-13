/*
*	TestUtils.cu
*/

#include "../lib/cppunit.h"

#include "../src/utils/utils.h"

class TestUtils : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestUtils);
	CPPUNIT_TEST(test1);	
	CPPUNIT_TEST_SUITE_END();

protected:
	void test1(void);

public:
	void setUp(void);
	void tearDown(void);

private:
	etmat S;
	etmat T;
	etmat U;
	etmat V;
	esmat W;
	etmat Z;

};

void
TestUtils::test1(void)
{
	CPPUNIT_ASSERT(true);
}

void TestUtils::setUp(void)
{
	T.push_back(Eigen::Triplet<double>({0,0,1}));
	T.push_back(Eigen::Triplet<double>({0,1,2}));
	T.push_back(Eigen::Triplet<double>({0,2,3}));
	T.push_back(Eigen::Triplet<double>({1,0,4}));
	T.push_back(Eigen::Triplet<double>({1,1,5}));
	T.push_back(Eigen::Triplet<double>({1,2,6}));
	T.push_back(Eigen::Triplet<double>({2,0,7}));
	T.push_back(Eigen::Triplet<double>({2,1,8}));
	T.push_back(Eigen::Triplet<double>({2,2,9}));

	S.push_back(Eigen::Triplet<double>({0,0,1}));
	S.push_back(Eigen::Triplet<double>({0,1,2}));
	S.push_back(Eigen::Triplet<double>({0,2,3}));
	S.push_back(Eigen::Triplet<double>({1,0,4}));
	S.push_back(Eigen::Triplet<double>({1,1,5}));
	S.push_back(Eigen::Triplet<double>({1,2,6}));
	S.push_back(Eigen::Triplet<double>({2,0,7}));
	S.push_back(Eigen::Triplet<double>({2,1,8}));
	S.push_back(Eigen::Triplet<double>({2,2,9}));
	
	std::cout << std::endl << "Sum" << std::endl;
	T += S;
	for(auto t : T)
	{
		std::cout << t.row()<< "x" << t.col() << ": " << t.value() << std::endl;
	}
	std::cout << std::endl << "Tranpose" << std::endl;
	U = transpose(T);
	for(auto t : U)
	{
		std::cout << t.row()<< "x" << t.col() << ": " << t.value() << std::endl;
	}
	W = esmat(3,3);
	W.setFromTriplets(T.begin(),T.end());
	std::cout << std::endl << "Second Column" << std::endl;
	V = esmat2etmat(getColumns(W,{1,2}));
	for(auto t : V)
	{
		std::cout << t.row()<< "x" << t.col() << ": " << t.value() << std::endl;
	}
	std::cout << std::endl << "Second Column with drift" << std::endl;
	Z = esmat2etmat(getColumns(W,{1}),5,10);
	std::cout << Z << std::endl;

	std::cout << getTimestamp() << std::endl;
}

void TestUtils::tearDown(void)
{
	//delete ;
}

CPPUNIT_TEST_SUITE_REGISTRATION( TestUtils );

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
	std::ofstream xmlFileOut("test/results/cppTestUtilsResults.xml");
	XmlOutputter xmlOut(&collectedresults, xmlFileOut);
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}


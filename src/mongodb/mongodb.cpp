/*
*	mongodb.cpp
*/

#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/json.hpp>
#include <bsoncxx/types.hpp>
#include <bsoncxx/types/value.hpp>

#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>

#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <bsoncxx/json.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/stdx.hpp>
#include <mongocxx/uri.hpp>

using bsoncxx::builder::stream::close_array;
using bsoncxx::builder::stream::close_document;
using bsoncxx::builder::stream::document;
using bsoncxx::builder::stream::finalize;
using bsoncxx::builder::stream::open_array;
using bsoncxx::builder::stream::open_document;

#include <stdexcept>
#include "dbconfig.h"
#include "../dmat/dmat.h"
#include "../utils/Mesh.h"
#include "struct.h"
#include "mongodb_impl.h"
#include "mongodb.h"

void drop(dbconfig db){
    mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{}};
    auto simCollection = conn[db.dbname]["sim"];
    auto timestepCollection = conn[db.dbname]["timestep"];
	auto plotDataCollection = conn[db.dbname]["plotData"];
	simCollection.drop();
	timestepCollection.drop();
	plotDataCollection.drop();
}

void save_sim(dbconfig db, miniSim sim){
    mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{}};
    auto simCollection = conn[db.dbname]["sim"];
	//sim.date = 
	if(simCollection.count( document{} << "id" << sim.id << finalize )==0)
	{
		simCollection.insert_one(sim2doc(sim).view());
	}
	else throw std::invalid_argument("save_sim: sim already present in collection!");
}

miniSim get_sim(dbconfig db, std::string id)
{
	mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{}};
    auto simCollection = conn[db.dbname]["sim"];
	mongocxx::cursor cursor = simCollection.find(
		document{} << "id" << id << finalize);
	miniSim s = doc2sim(*cursor.begin());
	return s;
}

int get_time(dbconfig db, std::string id)
{
	mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{}};
    auto timestepCollection = conn[db.dbname]["timestep"];
	int t = timestepCollection.count(
		document{} << "id" << id << finalize);
	return t;

}

void save_timestep(dbconfig db, timestep t)
{
std::cout << "Saving timestep id = " << t.id << ", time = " << t.time << std::endl; 
	mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{}};
    auto timestepCollection = conn[db.dbname]["timestep"];
	//t.date = 
	if(timestepCollection.count( document{} << "id" << t.id << "time" << t.time << finalize )==0)
	{
		timestepCollection.insert_one(timestep2doc(t).view());
	}
	else throw std::invalid_argument("save_timestep: timestep already present in collection!");
}

timestep get_timestep(dbconfig db, std::string id, int time)
{
	mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{}};
    auto timestepCollection = conn[db.dbname]["timestep"];

	mongocxx::cursor cursorts = timestepCollection.find(
		document{} << "id" << id << "time" << time << finalize);
	timestep t = doc2timestep(*cursorts.begin());
	return t;

}

void save_plotData(dbconfig db, plotData t)
{
std::cout << "Saving plotData id = " << t.id << ", time = " << t.time << std::endl; 
	mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{}};
    auto plotDataCollection = conn[db.dbname]["plotData"];
	//t.date = 
	if(plotDataCollection.count( document{} << "id" << t.id << "time" << t.time << finalize )==0)
	{
		plotDataCollection.insert_one(plotData2doc(t).view());
	}
	else throw std::invalid_argument("save_plotData: plotData already present in collection!");
}

plotData get_plotData(dbconfig db, std::string id, int time)
{
	mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{}};
    auto plotDataCollection = conn[db.dbname]["plotData"];

	mongocxx::cursor cursorts = plotDataCollection.find(
		document{} << "id" << id << "time" << time << finalize);
	plotData t = doc2plotData(*cursorts.begin());
	return t;

}


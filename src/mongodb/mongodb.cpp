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

	std::ostringstream ss;
	ss << db.dbname << " dropped.";
	LOG_OK(ss);
}

void save_sim(dbconfig db, miniSim sim){
    mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{}};
    auto simCollection = conn[db.dbname]["sim"];
	//sim.date = 
	if(simCollection.count( document{} << "id" << sim.id << finalize )==0)
	{
		simCollection.insert_one(sim2doc(sim).view());
		std::ostringstream ss;
		ss << "Saved simulation id = " << sim.id << ".";
		LOG_OK(ss);
	}
	else
	{
		std::ostringstream ss;
		ss << "save_sim: simulation id = " << sim.id << " already present in collection!";
		//LOG_ERROR(ss);
		//throw EXCEPTION("save_sim: sim already present in collection!");
		throw EXCEPTION(ss.str());
	}
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
	mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{}};
    auto timestepCollection = conn[db.dbname]["timestep"];
	//t.date = 
	if(timestepCollection.count( document{} << "id" << t.id << "time" << t.time << finalize )==0)
	{
		timestepCollection.insert_one(timestep2doc(t).view());
		std::ostringstream ss;
		ss << "Saved timestep id = " << t.id << ", time = " << t.time << ".";
		LOG_OK(ss);
	}
	else
	{
		std::ostringstream ss;
		ss << "save_timestep: timestep id = " << t.id << ", time = " << t.time << " already present in collection!";
		//LOG_ERROR(ss);
		//throw EXCEPTION("save_timestep: timestep already present in collection!");
		throw EXCEPTION(ss.str());
	}
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
	mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{}};
    auto plotDataCollection = conn[db.dbname]["plotData"];
	//t.date = 
	if(plotDataCollection.count( document{} << "id" << t.id << "time" << t.time << finalize )==0)
	{
		plotDataCollection.insert_one(plotData2doc(t).view());
		std::ostringstream ss;
		ss << "Saved timestep id = " << t.id << ", time = " << t.time << ".";
		LOG_OK(ss);
	}
	else
	{
		std::ostringstream ss;
		ss << "save_plotData: plotData id = " << t.id << ", time = " << t.time << " already present in collection!";
		//LOG_ERROR(ss);
		//throw EXCEPTION("save_plotData: plotData already present in collection!");
		throw EXCEPTION(ss.str());
	}
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


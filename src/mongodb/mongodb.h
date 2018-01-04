/*
*	mongodb.h
*/

#ifndef MONGODB_H_INCLUDED_
#define MONGODB_H_INCLUDED_

void drop(dbconfig db);
void save_sim(dbconfig db, miniSim sim);
miniSim get_sim(dbconfig db, std::string id);
int get_time(dbconfig db, std::string id);
void save_timestep(dbconfig db, timestep t);
timestep get_timestep(dbconfig db, std::string id,int time);
void save_plotData(dbconfig db, plotData t);
plotData get_plotData(dbconfig db, std::string id,int time);

#endif


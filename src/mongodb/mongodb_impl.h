/*
*	mongodb_impl.h
*/

#ifndef MONGODB_IMPL_H_INCLUDED_
#define MONGODB_IMPL_H_INCLUDED_

bsoncxx::builder::stream::document sim2doc(miniSim s);
miniSim doc2sim(bsoncxx::document::view doc);
document timestep2doc(timestep t);
timestep doc2timestep(bsoncxx::document::view doc);
document plotData2doc(plotData t);
plotData doc2plotData(bsoncxx::document::view doc);

#endif


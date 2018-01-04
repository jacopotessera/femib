/*
*	mongodb_impl.cpp
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

#include "../dmat/dmat.h"
#include "../utils/Mesh.h"
#include "struct.h"
#include "mongodb_impl.h"

bsoncxx::builder::stream::document sim2doc(miniSim s)
{
	// V
	bsoncxx::builder::stream::document docVP{};
	auto document_V_mesh_P = 
		docVP <<
		"P" << bsoncxx::builder::stream::open_array;
		for(int i=0;i<s.V.mesh.P.size();i++)
		{
			auto doc = document_V_mesh_P << bsoncxx::builder::stream::open_array;
			for(int j=0;j<s.V.mesh.P[i].size;j++)
			{
				doc << s.V.mesh.P[i](j);
			}
			doc << bsoncxx::builder::stream::close_array;
		}
		document_V_mesh_P << bsoncxx::builder::stream::close_array;

	bsoncxx::builder::stream::document docVT{};
	auto document_V_mesh_T = 
		docVT <<
		"T" << bsoncxx::builder::stream::open_array;
		for(int i=0;i<s.V.mesh.T.size();i++)
		{
			auto doc = document_V_mesh_T << bsoncxx::builder::stream::open_array;
			for(int j=0;j<s.V.mesh.T[i].size;j++)
			{
				doc << s.V.mesh.T[i](j);
			}
			doc << bsoncxx::builder::stream::close_array;
		}
		document_V_mesh_T << bsoncxx::builder::stream::close_array;


	bsoncxx::builder::stream::document docVE{};
	auto document_V_mesh_E = 
		docVE <<
		"E" << bsoncxx::builder::stream::open_array;
		for(int i=0;i<s.V.mesh.E.size();i++)
		{
			document_V_mesh_E << s.V.mesh.E[i];
		}
		document_V_mesh_E << bsoncxx::builder::stream::close_array;

	bsoncxx::builder::stream::document document_V_mesh{};
	document_V_mesh << "mesh" << bsoncxx::builder::stream::open_document
			<< bsoncxx::builder::concatenate_doc{docVP.view()}
			<< bsoncxx::builder::concatenate_doc{docVT.view()}
			<< bsoncxx::builder::concatenate_doc{docVE.view()}
		<< bsoncxx::builder::stream::close_document;

	bsoncxx::builder::stream::document document_V{};
	document_V 
		<< "V" << bsoncxx::builder::stream::open_document
			<< "finiteElement" << s.V.finiteElement
			<< "gauss" << s.V.gauss
			<< bsoncxx::builder::concatenate_doc{document_V_mesh.view()}
		<< bsoncxx::builder::stream::close_document;

	// Q
	bsoncxx::builder::stream::document docQP{};
	auto document_Q_mesh_P = 
		docQP <<
		"P" << bsoncxx::builder::stream::open_array;
		for(int i=0;i<s.Q.mesh.P.size();i++)
		{
			auto doc = document_Q_mesh_P << bsoncxx::builder::stream::open_array;
			for(int j=0;j<s.Q.mesh.P[i].size;j++)
			{
				doc << s.Q.mesh.P[i](j);
			}
			doc << bsoncxx::builder::stream::close_array;
		}
		document_Q_mesh_P << bsoncxx::builder::stream::close_array;

	bsoncxx::builder::stream::document docQT{};
	auto document_Q_mesh_T = 
		docQT <<
		"T" << bsoncxx::builder::stream::open_array;
		for(int i=0;i<s.Q.mesh.T.size();i++)
		{
			auto doc = document_Q_mesh_T << bsoncxx::builder::stream::open_array;
			for(int j=0;j<s.Q.mesh.T[i].size;j++)
			{
				doc << s.Q.mesh.T[i](j);
			}
			doc << bsoncxx::builder::stream::close_array;
		}
		document_Q_mesh_T << bsoncxx::builder::stream::close_array;


	bsoncxx::builder::stream::document docQE{};
	auto document_Q_mesh_E = 
		docQE <<
		"E" << bsoncxx::builder::stream::open_array;
		for(int i=0;i<s.Q.mesh.E.size();i++)
		{
			document_Q_mesh_E << s.Q.mesh.E[i];
		}
		document_Q_mesh_E << bsoncxx::builder::stream::close_array;

	bsoncxx::builder::stream::document document_Q_mesh{};
	document_Q_mesh << "mesh" << bsoncxx::builder::stream::open_document
			<< bsoncxx::builder::concatenate_doc{docQP.view()}
			<< bsoncxx::builder::concatenate_doc{docQT.view()}
			<< bsoncxx::builder::concatenate_doc{docQE.view()}
		<< bsoncxx::builder::stream::close_document;

	bsoncxx::builder::stream::document document_Q{};
	document_Q 
		<< "Q" << bsoncxx::builder::stream::open_document
			<< "finiteElement" << s.Q.finiteElement
			<< "gauss" << s.Q.gauss
			<< bsoncxx::builder::concatenate_doc{document_Q_mesh.view()}
		<< bsoncxx::builder::stream::close_document;

	// S
	bsoncxx::builder::stream::document docSP{};
	auto document_S_mesh_P = 
		docSP <<
		"P" << bsoncxx::builder::stream::open_array;
		for(int i=0;i<s.S.mesh.P.size();i++)
		{
			auto doc = document_S_mesh_P << bsoncxx::builder::stream::open_array;
			for(int j=0;j<s.S.mesh.P[i].size;j++)
			{
				doc << s.S.mesh.P[i](j);
			}
			doc << bsoncxx::builder::stream::close_array;
		}
		document_S_mesh_P << bsoncxx::builder::stream::close_array;

	bsoncxx::builder::stream::document docST{};
	auto document_S_mesh_T = 
		docST <<
		"T" << bsoncxx::builder::stream::open_array;
		for(int i=0;i<s.S.mesh.T.size();i++)
		{
			auto doc = document_S_mesh_T << bsoncxx::builder::stream::open_array;
			for(int j=0;j<s.S.mesh.T[i].size;j++)
			{
				doc << s.S.mesh.T[i](j);
			}
			doc << bsoncxx::builder::stream::close_array;
		}
		document_S_mesh_T << bsoncxx::builder::stream::close_array;

	bsoncxx::builder::stream::document docSE{};
	auto document_S_mesh_E = 
		docSE <<
		"E" << bsoncxx::builder::stream::open_array;
		for(int i=0;i<s.S.mesh.E.size();i++)
		{
			document_S_mesh_E << s.S.mesh.E[i];
		}
		document_S_mesh_E << bsoncxx::builder::stream::close_array;

	bsoncxx::builder::stream::document document_S_mesh{};
	document_S_mesh << "mesh" << bsoncxx::builder::stream::open_document
			<< bsoncxx::builder::concatenate_doc{docSP.view()}
			<< bsoncxx::builder::concatenate_doc{docST.view()}
			<< bsoncxx::builder::concatenate_doc{docSE.view()}
		<< bsoncxx::builder::stream::close_document;

	bsoncxx::builder::stream::document document_S{};
	document_S 
		<< "S" << bsoncxx::builder::stream::open_document
			<< "finiteElement" << s.S.finiteElement
			<< "gauss" << s.S.gauss
			<< bsoncxx::builder::concatenate_doc{document_S_mesh.view()}
		<< bsoncxx::builder::stream::close_document;

	// L
	bsoncxx::builder::stream::document docLP{};
	auto document_L_mesh_P = 
		docLP <<
		"P" << bsoncxx::builder::stream::open_array;
		for(int i=0;i<s.L.mesh.P.size();i++)
		{
			auto doc = document_L_mesh_P << bsoncxx::builder::stream::open_array;
			for(int j=0;j<s.L.mesh.P[i].size;j++)
			{
				doc << s.L.mesh.P[i](j);
			}
			doc << bsoncxx::builder::stream::close_array;
		}
		document_L_mesh_P << bsoncxx::builder::stream::close_array;

	bsoncxx::builder::stream::document docLT{};
	auto document_L_mesh_T = 
		docLT <<
		"T" << bsoncxx::builder::stream::open_array;
		for(int i=0;i<s.L.mesh.T.size();i++)
		{
			auto doc = document_L_mesh_T << bsoncxx::builder::stream::open_array;
			for(int j=0;j<s.L.mesh.T[i].size;j++)
			{
				doc << s.L.mesh.T[i](j);
			}
			doc << bsoncxx::builder::stream::close_array;
		}
		document_L_mesh_T << bsoncxx::builder::stream::close_array;


	bsoncxx::builder::stream::document docLE{};
	auto document_L_mesh_E = 
		docLE <<
		"E" << bsoncxx::builder::stream::open_array;
		for(int i=0;i<s.L.mesh.E.size();i++)
		{
			document_L_mesh_E << s.L.mesh.E[i];
		}
		document_L_mesh_E << bsoncxx::builder::stream::close_array;

	bsoncxx::builder::stream::document document_L_mesh{};
	document_L_mesh << "mesh" << bsoncxx::builder::stream::open_document
			<< bsoncxx::builder::concatenate_doc{docLP.view()}
			<< bsoncxx::builder::concatenate_doc{docLT.view()}
			<< bsoncxx::builder::concatenate_doc{docLE.view()}
		<< bsoncxx::builder::stream::close_document;

	bsoncxx::builder::stream::document document_L{};
	document_L 
		<< "L" << bsoncxx::builder::stream::open_document
			<< "finiteElement" << s.L.finiteElement
			<< "gauss" << s.L.gauss
			<< bsoncxx::builder::concatenate_doc{document_L_mesh.view()}
		<< bsoncxx::builder::stream::close_document;

	// doc
	bsoncxx::builder::stream::document document{};
	document
		<< "id" << s.id
		<< "date" << s.date
		<< "full" << s.full
		<< "parameters" << bsoncxx::builder::stream::open_document
			<< "rho" << s.parameters.rho
			<< "eta" << s.parameters.eta
			<< "deltarho" << s.parameters.deltarho
			<< "kappa" << s.parameters.kappa
			<< "deltat" << s.parameters.deltat
			<< "steps" << s.parameters.steps
			<< "TMAX" << s.parameters.TMAX
		<< bsoncxx::builder::stream::close_document
		<< bsoncxx::builder::concatenate_doc{document_V.view()}
		<< bsoncxx::builder::concatenate_doc{document_Q.view()}
		<< bsoncxx::builder::concatenate_doc{document_S.view()}
		<< bsoncxx::builder::concatenate_doc{document_L.view()};

	return document;
}

miniSim doc2sim(bsoncxx::document::view doc)
{
	miniSim s;

	s.id = doc["id"].get_utf8().value.to_string();
	s.date = doc["date"].get_utf8().value.to_string();
	s.full = doc["full"].get_bool();

	s.parameters.rho = doc["parameters"]["rho"].get_double();
	s.parameters.eta = doc["parameters"]["eta"].get_double();
	s.parameters.deltarho = doc["parameters"]["deltarho"].get_double();
	s.parameters.kappa = doc["parameters"]["kappa"].get_double();
	s.parameters.deltat = doc["parameters"]["deltat"].get_double();
	s.parameters.steps = doc["parameters"]["steps"].get_int32();
	s.parameters.TMAX = doc["parameters"]["TMAX"].get_double();

	for (auto i : doc["V"]["mesh"]["P"].get_array().value)
	{
		dvec v;
		for(auto j: i.get_array().value)
		{
			v.size += 1;
			v(v.size-1) = j.get_double();
		}

		s.V.mesh.P.push_back(v);
	}
	for (auto i : doc["V"]["mesh"]["T"].get_array().value)
	{
		ditrian v;
		for(auto j: i.get_array().value)
		{
			v.size += 1;
			v(v.size-1) = j.get_int32();
		}

		s.V.mesh.T.push_back(v);
	}
	for (auto i : doc["V"]["mesh"]["E"].get_array().value)
	{
		s.V.mesh.E.push_back(i.get_int32());
	}
	s.V.finiteElement = doc["V"]["finiteElement"].get_utf8().value.to_string();
	s.V.gauss = doc["V"]["gauss"].get_utf8().value.to_string();

	for (auto i : doc["Q"]["mesh"]["P"].get_array().value)
	{
		dvec v;
		for(auto j: i.get_array().value)
		{
			v.size += 1;
			v(v.size-1) = j.get_double();
		}

		s.Q.mesh.P.push_back(v);
	}
	for (auto i : doc["Q"]["mesh"]["T"].get_array().value)
	{
		ditrian v;
		for(auto j: i.get_array().value)
		{
			v.size += 1;
			v(v.size-1) = j.get_int32();
		}

		s.Q.mesh.T.push_back(v);
	}
	for (auto i : doc["Q"]["mesh"]["E"].get_array().value)
	{
		s.Q.mesh.E.push_back(i.get_int32());
	}
	s.Q.finiteElement = doc["Q"]["finiteElement"].get_utf8().value.to_string();
	s.Q.gauss = doc["Q"]["gauss"].get_utf8().value.to_string();

	for (auto i : doc["S"]["mesh"]["P"].get_array().value)
	{
		dvec v;
		for(auto j: i.get_array().value)
		{
			v.size += 1;
			v(v.size-1) = j.get_double();
		}
		s.S.mesh.P.push_back(v);
	}
	for (auto i : doc["S"]["mesh"]["T"].get_array().value)
	{
		ditrian v;
		for(auto j: i.get_array().value)
		{
			v.size += 1;
			v(v.size-1) = j.get_int32();
		}

		s.S.mesh.T.push_back(v);
	}
	for (auto i : doc["S"]["mesh"]["E"].get_array().value)
	{
		s.S.mesh.E.push_back(i.get_int32());
	}
	s.S.finiteElement = doc["S"]["finiteElement"].get_utf8().value.to_string();
	s.S.gauss = doc["S"]["gauss"].get_utf8().value.to_string();

	for (auto i : doc["L"]["mesh"]["P"].get_array().value)
	{
		dvec v;
		for(auto j: i.get_array().value)
		{
			v.size += 1;
			v(v.size-1) = j.get_double();
		}
		s.L.mesh.P.push_back(v);
	}
	for (auto i : doc["L"]["mesh"]["T"].get_array().value)
	{
		ditrian v;
		for(auto j: i.get_array().value)
		{
			v.size += 1;
			v(v.size-1) = j.get_int32();
		}
		s.L.mesh.T.push_back(v);
	}
	for (auto i : doc["L"]["mesh"]["E"].get_array().value)
	{
		s.L.mesh.E.push_back(i.get_int32());
	}
	s.L.finiteElement = doc["L"]["finiteElement"].get_utf8().value.to_string();
	s.L.gauss = doc["L"]["gauss"].get_utf8().value.to_string();

	return s;
}

document timestep2doc(timestep t)
{
	document data_builder{};
	data_builder
		<< "id" << t.id
		<< "time" << t.time;
	auto array_builder_u = data_builder
		<< "u" << open_array;
		for(int i=0;i<t.u.size();i++)
		{
			array_builder_u << t.u[i];
		}
		array_builder_u << close_array;
	auto array_builder_q = data_builder
		<< "q" << open_array;
		for(int i=0;i<t.q.size();i++)
		{
			array_builder_q << t.q[i];
		}
		array_builder_q << close_array;
	auto array_builder_x = data_builder
		<< "x" << open_array;
		for(int i=0;i<t.x.size();i++)
		{
			array_builder_x << t.x[i];
		}
		array_builder_x << close_array;
	auto array_builder_l = data_builder
		<< "l" << open_array;
		for(int i=0;i<t.l.size();i++)
		{
			array_builder_l << t.l[i];
		}
		array_builder_l << close_array;
	return data_builder;
}

timestep doc2timestep(bsoncxx::document::view doc)
{
	timestep s;	
	s.id = doc["id"].get_utf8().value.to_string();
	s.time = doc["time"].get_int32();
	for (auto i : doc["u"].get_array().value)
	{
		s.u.push_back(i.get_double());
	}
	for (auto i : doc["q"].get_array().value)
	{
		s.q.push_back(i.get_double());
	}
	for (auto i : doc["x"].get_array().value)
	{
		s.x.push_back(i.get_double());
	}
	for (auto i : doc["l"].get_array().value)
	{
		s.l.push_back(i.get_double());
	}
	return s;
}


document plotData2doc(plotData t)
{
	document data_builder{};
	data_builder
		<< "id" << t.id
		<< "time" << t.time;

	auto array_builder_u = data_builder
		<< "u" << open_array;
	for(int i=0;i<t.u.size();++i)
	{
		auto doc = array_builder_u << bsoncxx::builder::stream::open_array;
		for(int j=0;j<t.u[i].size();++j)
		{
			auto doc_ = doc << bsoncxx::builder::stream::open_array;
			for(int k=0;k<t.u[i][j].size();++k)
			{
				doc_ << t.u[i][j][k];
			}
			doc_ << bsoncxx::builder::stream::close_array;
		}
		doc << bsoncxx::builder::stream::close_array;
	}
	array_builder_u << close_array;

	auto array_builder_q = data_builder
		<< "q" << open_array;
	for(int i=0;i<t.q.size();++i)
	{
		auto doc = array_builder_q << bsoncxx::builder::stream::open_array;
		for(int j=0;j<t.q[i].size();++j)
		{
			auto doc_ = doc << bsoncxx::builder::stream::open_array;
			for(int k=0;k<t.q[i][j].size();++k)
			{
				doc_ << t.q[i][j][k];
			}
			doc_ << bsoncxx::builder::stream::close_array;
		}
		doc << bsoncxx::builder::stream::close_array;
	}
	array_builder_q << close_array;

	auto array_builder_x = data_builder
		<< "x" << open_array;
	for(int i=0;i<t.x.size();++i)
	{
		auto doc = array_builder_x << bsoncxx::builder::stream::open_array;
		for(int j=0;j<t.x[i].size();++j)
		{
			auto doc_ = doc << bsoncxx::builder::stream::open_array;
			for(int k=0;k<t.x[i][j].size();++k)
			{
				doc_ << t.x[i][j][k];
			}
			doc_ << bsoncxx::builder::stream::close_array;
		}
		doc << bsoncxx::builder::stream::close_array;
	}
	array_builder_x << close_array;

	return data_builder;
}

plotData doc2plotData(bsoncxx::document::view doc)
{
	plotData s;
	return s;
}


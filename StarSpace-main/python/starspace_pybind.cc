#include <spacexplore.h>
#include <matrix.h>
#include <model.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(starwrap, m) {
	py::class_<spacexplore::Args, std::shared_ptr<spacexplore::Args>>(m, "args")
		.def(py::init<>())
		.def_readwrite("trainFile", &spacexplore::Args::trainFile)
		.def_readwrite("validationFile", &spacexplore::Args::validationFile)
		.def_readwrite("testFile", &spacexplore::Args::testFile)
		.def_readwrite("predictionFile", &spacexplore::Args::predictionFile)
		.def_readwrite("model", &spacexplore::Args::model)
		.def_readwrite("initModel", &spacexplore::Args::initModel)
		.def_readwrite("fileFormat", &spacexplore::Args::fileFormat)
		.def_readwrite("label", &spacexplore::Args::label)
		.def_readwrite("basedoc", &spacexplore::Args::basedoc)
		.def_readwrite("loss", &spacexplore::Args::loss)
		.def_readwrite("similarity", &spacexplore::Args::similarity)
		.def_readwrite("lr", &spacexplore::Args::lr)
		.def_readwrite("termLr", &spacexplore::Args::termLr)
		.def_readwrite("norm", &spacexplore::Args::norm)
		.def_readwrite("margin", &spacexplore::Args::margin)
		.def_readwrite("initRandSd", &spacexplore::Args::initRandSd)
		.def_readwrite("p", &spacexplore::Args::p)
		.def_readwrite("dropoutLHS", &spacexplore::Args::dropoutLHS)
		.def_readwrite("dropoutRHS", &spacexplore::Args::dropoutRHS)
		.def_readwrite("wordWeight", &spacexplore::Args::wordWeight)
		.def_readwrite("dim", &spacexplore::Args::dim)
		.def_readwrite("epoch", &spacexplore::Args::epoch)
		.def_readwrite("ws", &spacexplore::Args::ws)
		.def_readwrite("maxTrainTime", &spacexplore::Args::maxTrainTime)
		.def_readwrite("validationPatience", &spacexplore::Args::validationPatience)
		.def_readwrite("thread", &spacexplore::Args::thread)
		.def_readwrite("maxNegSamples", &spacexplore::Args::maxNegSamples)
		.def_readwrite("negSearchLimit", &spacexplore::Args::negSearchLimit)
		.def_readwrite("minCount", &spacexplore::Args::minCount)
		.def_readwrite("minCountLabel", &spacexplore::Args::minCountLabel)
		.def_readwrite("bucket", &spacexplore::Args::bucket)
		.def_readwrite("ngrams", &spacexplore::Args::ngrams)
		.def_readwrite("trainMode", &spacexplore::Args::trainMode)
		.def_readwrite("K", &spacexplore::Args::K)
		.def_readwrite("batchSize", &spacexplore::Args::batchSize)
		.def_readwrite("verbose", &spacexplore::Args::verbose)
		.def_readwrite("debug", &spacexplore::Args::debug)
		.def_readwrite("adagrad", &spacexplore::Args::adagrad)
		.def_readwrite("isTrain", &spacexplore::Args::isTrain)
		.def_readwrite("normalizeText", &spacexplore::Args::normalizeText)
		.def_readwrite("saveEveryEpoch", &spacexplore::Args::saveEveryEpoch)
		.def_readwrite("saveTempModel", &spacexplore::Args::saveTempModel)
		.def_readwrite("shareEmb", &spacexplore::Args::shareEmb)
		.def_readwrite("useWeight", &spacexplore::Args::useWeight)
		.def_readwrite("trainWord", &spacexplore::Args::trainWord)
		.def_readwrite("excludeLHS", &spacexplore::Args::excludeLHS)
		;

	py::class_<spacexplore::Matrix <spacexplore::Real>>(m, "Matrix", py::buffer_protocol())
		.def_buffer([](spacexplore::Matrix <spacexplore::Real> &m) -> py::buffer_info {
			return py::buffer_info(
				m.matrix.data().begin(),							/* Pointer to buffer */
				sizeof(spacexplore::Real),							/* Size of one scalar */
				py::format_descriptor<spacexplore::Real>::format(), 	/* Python struct-style format descriptor */
				2,									    			/* Number of dimensions */
				{ m.numRows(), m.numCols() },				        /* Buffer dimensions */
				{ sizeof(spacexplore::Real) * m.numCols(),			/* Strides (in bytes) for each index */
				  sizeof(spacexplore::Real) }
			);
		}
	);

	py::class_<spacexplore::spacexplore>(m, "spacexplore")
		.def(py::init<std::shared_ptr<spacexplore::Args>>())
		.def("init", &spacexplore::spacexplore::init)
		.def("initFromTsv", &spacexplore::spacexplore::initFromTsv)
		.def("initFromSavedModel", &spacexplore::spacexplore::initFromSavedModel)

		.def("train", &spacexplore::spacexplore::train)
		.def("evaluate", &spacexplore::spacexplore::evaluate)

		.def("getDocVector", &spacexplore::spacexplore::getDocVector)

		.def("nearestNeighbor", &spacexplore::spacexplore::nearestNeighbor)
		.def("predictTags", &spacexplore::spacexplore::predictTags)

		.def("saveModel", &spacexplore::spacexplore::saveModel)
		.def("saveModelTsv", &spacexplore::spacexplore::saveModelTsv)
		.def("loadBaseDocs", &spacexplore::spacexplore::loadBaseDocs)
		;
}

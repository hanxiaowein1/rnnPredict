#include "ar_py.h"


AR::AR(std::string config_path)
{
	init(config_path);
}

AR::~AR()
{
	free();
}

void AR::init(std::string config_path)
{
	//ar_handle = initialize_handle(config_path);
	ar_handle =(ArHandle) new ModelHolder(config_path);
}

void AR::free()
{
	//freeModelMem(ar_handle);
	auto model_holder = (ModelHolder*)ar_handle;
	delete model_holder;
}

void AR::process(
	std::vector<Anno>& annos, py::array_t<unsigned char> img, 
	int rows, int cols, double mpp, int n, double score_threshold, double remove_threshold)
{
	cv::Mat mat(rows, cols, CV_8UC3, (uchar*)img.data());
	::process(annos, ar_handle, mat, mpp, n, score_threshold, remove_threshold);
}

PYBIND11_MAKE_OPAQUE(std::vector<Anno>);

PYBIND11_MODULE(ar, m)
{
	py::class_<Anno>(m, "Anno")
		.def(py::init<>())
		.def_readwrite("id", &Anno::id)
		.def_readwrite("x", &Anno::x)
		.def_readwrite("y", &Anno::y)
		.def_readwrite("type", &Anno::type)
		.def_readwrite("score", &Anno::score);
	py::bind_vector<std::vector<Anno>>(m, "VectorAnno");

	py::class_<AR>(m, "AR")
		.def(py::init<std::string>())
		.def("process", &AR::process);

}
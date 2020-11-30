#pragma once

#ifndef _AILAB_ARPY_H_
#define _AILAB_ARPY_H_

#include "ar.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;

class AR {
private:
	ArHandle ar_handle;
	void init(std::string config_path);
	void free();
public:
	AR() = delete;
	AR(std::string config_path);
	~AR();
	void process(std::vector<Anno>& annos, py::array_t<unsigned char> img, int rows, int cols, double mpp, int n);
};


#endif
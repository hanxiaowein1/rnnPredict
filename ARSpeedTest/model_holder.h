#pragma once

#ifndef _MODELHOLDER_H_
#define _MODELHOLDER_H_

#include "TrModel1.h"
#include "TrModel2.h"
#include "TfModel1.h"
#include "TfModel2.h"
#include "IniConfig.h"

typedef unsigned long long ArHandle;


struct TrModelHolder {
	TrModel1* tr_model1 = nullptr;
	TrModel2* tr_model2 = nullptr;
	bool isValid() {
		if (tr_model1 == nullptr || tr_model2 == nullptr) {
			return false;
		}
		else {
			return true;
		}
	}
};

struct TfModelHolder {
	TfModel1* tf_model1 = nullptr;
	TfModel2* tf_model2 = nullptr;
	bool isValid() {
		if (tf_model1 == nullptr || tf_model2 == nullptr) {
			return false;
		}
		else {
			return true;
		}
	}
};



//小小的封装一下，不然很难调用
class ModelHolder {
public:
	ArHandle handle = 0;
	ModelHolder(std::string ini_path) {
		setIniPath(ini_path.c_str());
		if (IniConfig::instance().getIniString("TensorRT", "USE_TR") == "OFF") {
			//使用tensorflow
			TfModelHolder* tf_model_holder = new TfModelHolder;
			tf_model_holder->tf_model1 = new TfModel1("TfModel1");
			tf_model_holder->tf_model1->createThreadPool(1);
			tf_model_holder->tf_model2 = new TfModel2("TfModel2");
			tf_model_holder->tf_model2->createThreadPool(1);
			handle = ArHandle(tf_model_holder);
		}
		else {
			TrModelHolder* tr_model_holder = new TrModelHolder;
			tr_model_holder->tr_model1 = new TrModel1("TrModel1");
			tr_model_holder->tr_model1->createThreadPool(1);
			tr_model_holder->tr_model2 = new TrModel2("TrModel2");
			tr_model_holder->tr_model2->createThreadPool(1);
			handle = ArHandle(tr_model_holder);
		}
	}

	~ModelHolder() {
		if (IniConfig::instance().getIniString("TensorRT", "USE_TR") == "OFF") {
			auto tf_model_holder = (TfModelHolder*)handle;
			delete tf_model_holder->tf_model1;
			tf_model_holder->tf_model1 = nullptr;
			delete tf_model_holder->tf_model2;
			tf_model_holder->tf_model2 = nullptr;
			delete tf_model_holder;
			tf_model_holder = nullptr;
		}
		else {
			auto tr_model_holder = (TrModelHolder*)handle;
			delete tr_model_holder->tr_model1;
			tr_model_holder->tr_model1 = nullptr;
			delete tr_model_holder->tr_model2;
			tr_model_holder->tr_model2 = nullptr;
			delete tr_model_holder;
			tr_model_holder = nullptr;
		}
	}

	void processDataConcurrencyM1(std::vector<cv::Mat>& imgs) {
		if (IniConfig::instance().getIniString("TensorRT", "USE_TR") == "OFF") {
			auto tf_model_holder = (TfModelHolder*)handle;
			tf_model_holder->tf_model1->processDataConcurrency(imgs);
		}
		else {
			auto tr_model_holder = (TrModelHolder*)handle;
			tr_model_holder->tr_model1->processDataConcurrency(imgs);
		}
	}

	void processDataConcurrencyM2(std::vector<cv::Mat>& imgs) {
		if (IniConfig::instance().getIniString("TensorRT", "USE_TR") == "OFF") {
			auto tf_model_holder = (TfModelHolder*)handle;
			tf_model_holder->tf_model2->processDataConcurrency(imgs);
		}
		else {
			auto tr_model_holder = (TrModelHolder*)handle;
			tr_model_holder->tr_model2->processDataConcurrency(imgs);
		}
	}

	std::vector<model1Result> getModel1Result() {
		if (IniConfig::instance().getIniString("TensorRT", "USE_TR") == "OFF") {
			auto tf_model_holder = (TfModelHolder*)handle;
			return tf_model_holder->tf_model1->m_results;
		}
		else {
			auto tr_model_holder = (TrModelHolder*)handle;
			return tr_model_holder->tr_model1->m_results;
		}
	}

	std::vector<model2Result> getModel2Result() {
		if (IniConfig::instance().getIniString("TensorRT", "USE_TR") == "OFF") {
			auto tf_model_holder = (TfModelHolder*)handle;
			return tf_model_holder->tf_model2->m_results;
		}
		else {
			auto tr_model_holder = (TrModelHolder*)handle;
			return tr_model_holder->tr_model2->m_results;
		}
	}

};

#endif

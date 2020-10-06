#ifndef _COMMONFUNC_H_
#define _COMMONFUNC_H_


#include <string>
#include <iostream>
#include <vector>
#include <io.h>
#include "TfBase.h"

using namespace std;

void saveAsTxt(vector<float> &score2, string name);
void splitString(string str, string c, vector<string> *outStr);
string getFileNamePrefix(string *path);
void getFiles(string path, vector<string> &files, string suffix);
void showTensor(tensorflow::Tensor &tensor);
void createDirRecursive(string dir);
string getFileName(string path);
void filterList(vector<string> &solveList, vector<string> &solvedList);
string getFileNameSuffix(string path);
#endif
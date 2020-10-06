#include "commonFunction.h"


void saveAsTxt(vector<float> &score2, string name) {
	FILE *file = fopen(name.c_str(), "w+");
	if (file == NULL)
	{
		printf("the path may unexist, create new files in local\n");
		file = fopen(("./" + name).c_str(), "w+");
		if (file == NULL)
		{
			printf("cannot create new file in local\n");
			return;
		}
		else
		{
			for (int i = 0; i < score2.size(); i++)
			{
				fprintf(file, "%f\n", score2[i]);
			}
			fclose(file);
			return;
		}
	}
	else
	{
		for (int i = 0; i < score2.size(); i++)
		{
			fprintf(file, "%f\n", score2[i]);
		}
		fclose(file);
	}
}

void splitString(string str, string c, vector<string> *outStr) {
	int pos = 0;
	while (str.find(c) != -1) {
		pos = str.find(c);
		string tmp = str.substr(0, pos);
		(*outStr).push_back(tmp);
		str = str.substr(pos + c.size());
	}
	(*outStr).push_back(str);
}

//ͨ��·���õ�һ���ļ���ǰ׺��
string getFileNamePrefix(string *path)
{
	if ((*path) == "") {
		cout << "getFileNamePrefix: the path should not be null!" << endl;
		return "";
	}
	vector<string> pathSplit;
	splitString(*path, "\\", &pathSplit);
	//���к�׺�����ļ���
	string imgName = pathSplit[pathSplit.size() - 1];
	//�ֽ��.
	vector<string> imgSplit;
	splitString(imgName, ".", &imgSplit);
	if (imgSplit.size() <= 1)
	{
		cout << "get file name prefix suffix, please check file name again\n";
		return "";
	}
		
	string prefix = "";
	for (int i = 0; i < imgSplit.size() - 2; i++)
	{
		prefix = prefix + imgSplit[i] + ".";
	}
	prefix = prefix + imgSplit[imgSplit.size() - 2];
	return prefix;
}

string getFileNameSuffix(string path)
{
	if (path == "")
	{
		cout << "getFileNameSuffix: the path should not be null\n";
		return "";
	}
	vector<string> pathSplit;
	splitString(path, "\\", &pathSplit);
	string fileName = pathSplit[pathSplit.size() - 1];
	vector<string> nameSplit;
	splitString(fileName, ".", &nameSplit);
	string suffix = nameSplit[nameSplit.size() - 1];
	return suffix;
}

void getFiles(string path, vector<string> &files, string suffix)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
		do {
			//�����Ŀ¼,����֮  
			//�������,�����б�  
			if ((fileinfo.attrib &  _A_SUBDIR)) {
				//�����Ŀ¼����ʲôҲ����
				/*if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				getFiles(p.assign(path).append("\\").append(fileinfo.name), files, suffix);*/
			}
			else {
				string tempFilename = string(fileinfo.name);
				//���������޸ģ�ʹ���ܹ���Ӧ�κγ��ȵĺ�׺��
				std::size_t pointFound = tempFilename.find_last_of(".");//�������һ��"."��λ��
				std::size_t length = tempFilename.length();
				size_t suffixLen = length - (pointFound + 1);
				string suffixGet = tempFilename.substr(pointFound + 1, suffixLen);
				if (suffixGet == suffix) {
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void showTensor(tensorflow::Tensor &tensor)
{
	int dims = tensor.dims();
	cout << "dims is: " << dims;
	for (int i = 0; i < dims; i++)
	{
		cout << "dims[" << i << "]:" << tensor.dim_size(i) << " ";
	}
}
void createDirRecursive(string dir)
{
	if (dir == "")
	{
		cout << "create dir get null path!\n";
	}
	vector<string> pathSplit;
	splitString(dir, "\\", &pathSplit);
	string path = pathSplit[0];
	for (int i = 1; i < pathSplit.size(); i++)
	{
		path = path + "\\" + pathSplit[i];
		DWORD dirType = GetFileAttributesA(path.c_str());
		if (dirType == INVALID_FILE_ATTRIBUTES) {
			//��������ڣ��򴴽��ļ���
			bool flag = CreateDirectoryA(path.c_str(), NULL);
			if (!flag) {
				cout << "create savePath failed" << endl;
			}
		}
	}
}

//ͨ��·���õ��ļ���
string getFileName(string path)
{
	if (path == "") {
		cout << "getFileNamePrefix: the path should not be null!" << endl;
		return "";
	}
	vector<string> pathSplit;
	splitString(path, "\\", &pathSplit);
	string fileName = pathSplit[pathSplit.size() - 1];
	return fileName;
}


/*
**ǰһ����sdpc���ļ���·������һ����xml�ļ���·��
**���˵�ǰһ���ļ��б��кͺ�һ��xml�ļ��б���ͬ�����ļ�
*/
void filterList(vector<string> &solveList, vector<string> &solvedList)
{
	vector<string> solveName;
	for (int i = 0; i < solveList.size(); i++) {
		solveName.emplace_back(getFileName(solveList[i]));
	}

	vector<string> solveNamePre;
	vector<string> solvedNamePre;
	for (int i = 0; i < solveList.size(); i++) {
		solveNamePre.emplace_back(getFileNamePrefix(&solveList[i]));
	}
	for (int i = 0; i < solvedList.size(); i++) {
		solvedNamePre.emplace_back(getFileNamePrefix(&solvedList[i]));
	}

	vector<int> undeletedIndex;
	for (int i = 0; i < solveNamePre.size(); i++) {
		bool flag = true;//��Ϊtrue����������
		for (int j = 0; j < solvedNamePre.size(); j++) {
			if (solveNamePre[i].compare(solvedNamePre[j]) == 0) {
				flag = false;
				break;
			}
		}
		if (flag) {
			undeletedIndex.emplace_back(i);
		}
	}

	for (int i = 0; i < undeletedIndex.size(); i++)
	{
		solveList[i] = solveList[undeletedIndex[i]];
	}
	solveList.resize(undeletedIndex.size());
}

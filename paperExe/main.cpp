#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[])
{
    std::string filepath = "D:\\TEST_OUTPUT\\rnnPredict\\result_correction\\100_1169115 0893146.txt";
    std::string slidePath = "";
    std::ifstream file(filepath);
    std::string str;
    while (std::getline(file, str))
    {
        // Process str
        std::cout << str << std::endl;
    }
    system("pause");
    return 0;
}
#pragma once
#ifndef _AILAB_INICONFIG_H_
#define _AILAB_INICONFIG_H_

#include <string>

void setIniPath(std::string inipath);
class IniConfig
{
private:
	IniConfig();
	~IniConfig();
	//std::string m_inipath;
public:
	IniConfig& operator=(const IniConfig&) = delete;
	IniConfig& operator=(IniConfig&&) = delete;
	static IniConfig& instance();
	std::string getIniString(std::string group, std::string key);
	int getIniInt(std::string group, std::string key);
	double getIniDouble(std::string group, std::string key);
};

#endif
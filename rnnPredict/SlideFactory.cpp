#include "SlideFactory.h"

SlideFactory::SlideFactory()
{
}


SlideFactory::~SlideFactory()
{
}

unique_ptr<SlideRead> SlideFactory::createSlideProduct(const char* slidePath)
{
	unique_ptr<SlideRead> uPtr;
	std::string slidePathStr(slidePath);
	//获得后缀名
	string suffix = getFileNameSuffix(slidePathStr);
	if (suffix == "sdpc")
	{
		uPtr.reset(new SdpcSlideRead(slidePath));
	}
	else if (suffix == "srp")
	{
		uPtr.reset(new SrpSlideRead(slidePath));
	}
	else
	{
		uPtr.reset(new OpenSlideRead(slidePath));
	}
	return uPtr;
}
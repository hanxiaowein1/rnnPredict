#include "OpenSlideRead.h"



OpenSlideRead::OpenSlideRead()
{
}

OpenSlideRead::OpenSlideRead(const char *slidePath):SlideRead()
{
	iniHandle(slidePath);
	ini_ration();
	getSlideBoundX(m_boundX);
	getSlideBoundY(m_boundY);
}

OpenSlideRead::~OpenSlideRead()
{
	openslide_close(osr);
}

void OpenSlideRead::iniHandle(const char * slidePath)
{
	osr = openslide_open(slidePath);
}

double OpenSlideRead::get_os_property_double(openslide_t* slide, const char* propName)
{
	const char* property = openslide_get_property_value(slide, propName);
	if (property == NULL)
	{
		return -1;
	}
	double ret = atof(property);
	return ret;
}

int OpenSlideRead::get_os_property(openslide_t *slide, const char* propName)
{
	const char *property = openslide_get_property_value(slide, propName);
	if (property == NULL) {
		return 0;
	}

	std::stringstream strValue;
	strValue << property;
	int intValue;
	strValue >> intValue;

	return intValue;
}

void OpenSlideRead::getSlideWidth(int & width)
{
	width = get_os_property(osr, "openslide.bounds-width");
	if (width == 0)
	{
		int height = 0;
		getLevelDimensions(0, width, height);//������getLevelDimensions��ȡwidth
	}
}

void OpenSlideRead::getSlideHeight(int & height)
{
	height = get_os_property(osr, "openslide.bounds-height");
	if (height == 0)
	{
		int width = 0;
		getLevelDimensions(0, width, height);//������getLevelDimensions��ȡwidth
	}
}

void OpenSlideRead::getSlideBoundX(int & boundX)
{
	boundX = get_os_property(osr, "openslide.bounds-x");
	m_boundX = boundX;
}

void OpenSlideRead::getSlideBoundY(int & boundY)
{
	boundY = get_os_property(osr, "openslide.bounds-y");
	m_boundY = boundY;
}

void OpenSlideRead::getSlideMpp(double & mpp)
{
	mpp = get_os_property_double(osr, "openslide.mpp-x");//��openslide���ĵ�������mppx��mppy��������mppx����mpp
	if (mpp == -1)
		mpp = 0.293f;
}

void OpenSlideRead::getLevelDimensions(int level, int & width, int & height)
{
	int64_t width64 = 0;
	int64_t height64 = 0;
	openslide_get_level_dimensions(osr, level, &width64, &height64);
	width = width64;
	height = height64;
}

bool OpenSlideRead::status()
{
	if(osr == nullptr)
		return false;
	return true;
}

void OpenSlideRead::getTile(int level, int x, int y, int width, int height, cv::Mat &img)
{
	if (width == 0 || height == 0)
		return;
	bufferManage(width, height, 4);
	//OpenSlide��read_region�е���֣�������ʼ��x, y����level0�µģ��Ժ��Ҷ����贫���x, y���ڶ�Ӧlevel�µ����꣬�����ǲ�����bound������
	//��ô����Ҫת������(�Ƚ�xתΪlevel0�µ�����)
	x = x * std::pow(m_ratio, level);
	y = y * std::pow(m_ratio, level);
	x = x + m_boundX;
	y = y + m_boundY;
	openslide_read_region(osr, (uint32_t*)buffer, x, y, level, width, height);
	cv::Mat image = cv::Mat(height, width, CV_8UC4, buffer, cv::Mat::AUTO_STEP).clone();
	cvtColor(image, img, cv::COLOR_RGBA2RGB);//Ҫת��һ��ͨ��
}

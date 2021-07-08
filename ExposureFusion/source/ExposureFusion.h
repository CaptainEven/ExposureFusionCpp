#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <io.h>
#include <fstream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <vector>
#include <time.h>
#include <windows.h>

using namespace cv;
using namespace std;

#define GRAY 1
#define COLOR1 2
#define COLOR2 3

#define MODE GRAY

#define BLOCKROWS 32
#define BLOCKCOLS 24


class QualityMeasures
{
private:
	Mat m_contrast;
	Mat m_saturation;
	Mat m_well_exposureness;
	Mat WeightMap;
	Mat WeightMapColor[3];  // B, G, R

	double GAUSS_WEIGHT[256] = { 0.0439369, 0.0461355, 0.0484255, 0.0508095, 0.0532905, 0.0558711, 0.0585542, 0.0613425, 0.0642389, 0.0672462, 0.0703673, 0.0736049, 0.0769618, 0.0804409, 0.084045, 0.0877768, 0.091639, 0.0956345, 0.0997657, 0.104035, 0.108446, 0.113, 0.117701, 0.122549, 0.127549, 0.132701, 0.138008, 0.143473, 0.149096, 0.154881, 0.160828, 0.166939, 0.173215, 0.179659, 0.186271, 0.193051, 0.200002, 0.207123, 0.214415, 0.221879, 0.229514, 0.237321, 0.245299, 0.253448, 0.261767, 0.270255, 0.27891, 0.287733, 0.29672, 0.305871, 0.315182, 0.324652, 0.334279, 0.344058, 0.353987, 0.364063, 0.374282, 0.384639, 0.395131, 0.405754, 0.416502, 0.42737, 0.438353, 0.449446, 0.460642, 0.471935, 0.48332, 0.494789, 0.506336, 0.517952, 0.529632, 0.541367, 0.553149, 0.56497, 0.576822, 0.588697, 0.600585, 0.612477, 0.624365, 0.636239, 0.648089, 0.659906, 0.671681, 0.683403, 0.695062, 0.706648, 0.718152, 0.729562, 0.740868, 0.752061, 0.763129, 0.774062, 0.78485, 0.795483, 0.805949, 0.81624, 0.826344, 0.836252, 0.845953, 0.855438, 0.864696, 0.873719, 0.882497, 0.89102, 0.89928, 0.907267, 0.914974, 0.922391, 0.929511, 0.936326, 0.942828, 0.94901, 0.954866, 0.960389, 0.965572, 0.97041, 0.974898, 0.979029, 0.982801, 0.986207, 0.989245, 0.991911, 0.994202, 0.996115, 0.997648, 0.998799, 0.999568, 0.999952, 0.999952, 0.999568, 0.998799, 0.997648, 0.996115, 0.994202, 0.991911, 0.989245, 0.986207, 0.9828, 0.979029, 0.974898, 0.97041, 0.965572, 0.960389, 0.954866, 0.94901, 0.942828, 0.936326, 0.929511, 0.922391, 0.914974, 0.907267, 0.89928, 0.89102, 0.882497, 0.873719, 0.864696, 0.855438, 0.845953, 0.836252, 0.826344, 0.81624, 0.805949, 0.795483, 0.78485, 0.774062, 0.763129, 0.75206, 0.740868, 0.729562, 0.718152, 0.706648, 0.695062, 0.683403, 0.671681, 0.659906, 0.648089, 0.636238, 0.624365, 0.612477, 0.600584, 0.588697, 0.576822, 0.56497, 0.553149, 0.541367, 0.529632, 0.517952, 0.506336, 0.494789, 0.48332, 0.471935, 0.460642, 0.449446, 0.438353, 0.42737, 0.416502, 0.405754, 0.395131, 0.384639, 0.374282, 0.364063, 0.353987, 0.344058, 0.334279, 0.324652, 0.315182, 0.305871, 0.29672, 0.287733, 0.27891, 0.270255, 0.261767, 0.253448, 0.245299, 0.237321, 0.229514, 0.221879, 0.214415, 0.207123, 0.200002, 0.193051, 0.18627, 0.179659, 0.173215, 0.166939, 0.160828, 0.154881, 0.149096, 0.143473, 0.138008, 0.132701, 0.127549, 0.122549, 0.117701, 0.113, 0.108446, 0.104035, 0.0997657, 0.0956345, 0.091639, 0.0877768, 0.084045, 0.0804409, 0.0769618, 0.0736048, 0.0703673, 0.0672462, 0.0642389, 0.0613425, 0.0585542, 0.0558711, 0.0532905, 0.0508095, 0.0484255, 0.0461355 };

public:

	QualityMeasures(const Mat& img_color, const Mat& img_gray)
	{
#if MODE==GRAY
		int ret = this->getContrastMeasure(img_gray, this->m_contrast);
		ret = this->getSaturationMeasure(img_color, this->m_saturation);
		ret = this->getWellExposednessMeasure(img_gray, this->m_well_exposureness);
		ret = this->getWeightMapImage(this->WeightMap);
#endif
	}

	int getContrastMeasure(const Mat& src, Mat& contrast);
	int getSaturationMeasure(const Mat& src, Mat& Saturation);
	int getWellExposednessMeasure(const Mat& src, Mat& well_exposure);
	int getWeightMapImage(Mat& weight_map);

	inline const Mat& getContrast() const
	{
		return this->m_contrast;
	}

	inline const Mat& getSaturation() const
	{
		return this->m_saturation;
	}

	inline const Mat& getWellExposureness() const
	{
		return this->m_well_exposureness;
	}

	inline const Mat& getWeightMap() const
	{
		return this->WeightMap;
	}

	inline const Mat& getWeightMapColor(const int nch) const
	{
		return this->WeightMapColor[nch];
	}
};


class ExposureFusion
{
private:
	vector<Mat> m_imgs_color;
	vector<Mat> m_imgs_gray;
	vector<Mat> m_weightMaps;
	vector<vector<Mat>> m_weightMapsColor;
	vector<Mat> m_normWeightMaps;
	vector<vector<Mat>> m_normWeightMapsColor;

	Mat m_resultImage;
	int m_nframes;
	int m_state;


public:
	// 构造函数
	ExposureFusion(const char* seq_path, const bool do_resize);

	//获取特定格式的文件名    
	const int getFilesFormat(const string& path, const string& format, vector<string>& files);

	void qualityMeasuresProcessing(void);
	// m_nframes weight map;

	// 处理过程
	bool saveImageBMP(const char* filename);

	void fuse(void);

	void setNormalizedWeightMaps();

	int setResultByPyramid(const int nch, Mat& channel);

	// 返回查询结果
	inline const cv::Mat& getResultImage()
	{
		return this->m_resultImage;
	}

	inline int getnframes() const
	{
		return this->m_nframes;
	}

	inline int getState() const
	{
		return this->m_state;
	}
};
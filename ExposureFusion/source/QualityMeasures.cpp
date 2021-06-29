#include "ExposureFusion.h"

using namespace cv;


int QualityMeasures::getContrastMeasure(const Mat& src, Mat& contrast)
{
	if (src.empty())
	{
		return -1;
	}

	Mat dst(src.size(), CV_16UC1);
	Mat lptemp(src.size(), CV_16SC1);
	cv::Laplacian(src, lptemp, CV_16SC1, 3);

	contrast.create(src.size(), CV_8UC1);  // 初始化内存空间
	for (int y = 0; y < contrast.rows; y++)
	{
		for (int x = 0; x < contrast.cols; x++)
		{
			contrast.at<uchar>(y, x) = (lptemp.at<short>(y, x) < 0) ? -lptemp.at<short>(y, x) : lptemp.at<short>(y, x);
		}
	}

	return 0;
}


int QualityMeasures::getSaturationMeasure(const Mat& src, Mat& saturation)
{
	if (src.empty())
	{
		return -1;
	}

	Mat std_dev_img(src.size(), CV_8UC1, CV_RGB(0, 0, 0));
	saturation = std_dev_img;

	const int nch = src.channels();
	float mean = 0;
	float variance = 0;
	float stdDev = 0;

#if MODE==GRAY
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			mean = (src.at<Vec3b>(y, x)[0] + src.at<Vec3b>(y, x)[1] + src.at<Vec3b>(y, x)[2]) / 3.0f;
			variance = 0;
			for (int i = 0; i < nch; i++)
			{
				variance += (mean - src.at<Vec3b>(y, x)[i]) * (mean - src.at<Vec3b>(y, x)[i]) / 3.0f;
			}
			stdDev = sqrt(variance);
			std_dev_img.at<uchar>(y, x) = (uchar)(stdDev + 0.5f);
		}
	}
#endif

	return 0;
}

int QualityMeasures::getWellExposednessMeasure(const Mat& src, Mat& well_exposure)
{
	if (src.empty())
	{
		return -1;
	}

	Mat dst(src.size(), CV_32FC1);
	well_exposure.create(src.size(), CV_32FC1);

	float new_pix_val = 0;
	float gauss_curve_weight = 0;

#if MODE==GRAY
	Mat temp(src.size(), CV_32FC1);
	Mat gray(src.size(), CV_8UC1);

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{

			gauss_curve_weight = (float)LUTWEN[src.at<uchar>(y, x)];
			new_pix_val = gauss_curve_weight * (float)src.at<uchar>(y, x);
			temp.at<float>(y, x) = new_pix_val;
			well_exposure.at<float>(y, x) = new_pix_val;
		}
	}

#endif

	return 0;
}

int QualityMeasures::getWeightMapImage(Mat& weight_map)
{
	const Mat& C = this->getContrast();
	const Mat& S = this->getSaturation();

	float pix = 0.0f;

	// float
	weight_map.create(C.size(), CV_32SC1);
	for (int y = 0; y < C.rows; y++)
	{
		for (int x = 0; x < C.cols; x++)
		{
			pix = float(C.at<uchar>(y, x) * S.at<uchar>(y, x));
			weight_map.at<int>(y, x) = int(pix);
		}
	}

	return 0;
}


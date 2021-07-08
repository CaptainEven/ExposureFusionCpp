#include "ExposureFusion.h"

using namespace cv;


int QualityMeasures::getContrastMeasure(const Mat& src, Mat& contrast)
{
	if (src.empty())
	{
		return -1;
	}

	contrast = Mat::zeros(src.size(), CV_32FC1);
	cv::Laplacian(src, contrast, -1, 3);

	// ----- normalize
	double min_v = 0.0;
	double max_v = 0.0;
	double* min_p = &min_v;
	double* max_p = &max_v;
	cv::minMaxLoc(contrast, min_p, max_p);

	Mat normalized;
	cv::normalize(contrast, normalized, 1.0, 0.0, cv::NORM_MINMAX);
	contrast = normalized;  // CV_8U

	return 0;
}


int QualityMeasures::getSaturationMeasure(const Mat& src, Mat& saturation)
{
	if (src.empty())
	{
		return -1;
	}

	saturation = Mat::zeros(src.size(), CV_32FC1);

	const int nch = src.channels();
	assert(nch == 3 or nch == 4);

	float mean = 0.0f;
	float variance = 0.0f;
	float std_dev = 0.0f;

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{ 
			// compute mean of RGB in each pixel
			const UINT8& b = src.at<Vec3b>(y, x)[0];
			const UINT8& g = src.at<Vec3b>(y, x)[1];
			const UINT8& r = src.at<Vec3b>(y, x)[2];

			mean = (float(b) + float(g) + float(r)) / 3.0f;

			// compute variance and std of RGB in each pixel
			variance = 0.0f;
			variance += (mean - float(b)) * (mean - float(b));
			variance += (mean - float(g)) * (mean - float(g));
			variance += (mean - float(r)) * (mean - float(r));
			variance /= 3.0f;

			std_dev = sqrtf(variance);
			saturation.at<float>(y, x) = std_dev;
		}
	}

	return 0;
}

int QualityMeasures::getWellExposednessMeasure(const Mat& src, Mat& well_exposure)
{
	if (src.empty())  // gray image
	{
		return -1;
	}

	well_exposure = Mat::zeros(src.size(), CV_32FC1);

	float new_pix_val = 0;
	float gauss_weight = 0;

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{

			gauss_weight = (float)GAUSS_WEIGHT[src.at<uchar>(y, x)];
			new_pix_val = gauss_weight * (float)src.at<uchar>(y, x);
			well_exposure.at<float>(y, x) = new_pix_val;
		}
	}

	return 0;
}

int QualityMeasures::getWeightMapImage(Mat& weight_map)
{
	const Mat& C = this->getContrast();          // uint8
	const Mat& S = this->getSaturation();        // float32
	const Mat& E = this->getWellExposureness();  // float32

	float pix = 0.0f;

	// float
	weight_map = Mat::zeros(C.size(), CV_32FC1);
	for (int y = 0; y < C.rows; y++)
	{
		for (int x = 0; x < C.cols; x++)
		{
			const uchar& C_weight = C.at<uchar>(y, x);
			const float& S_weight = S.at<float>(y, x);
			const float& E_weight = E.at<float>(y, x);
			pix = float(C_weight) * S_weight * E_weight + float(1e-12);
			weight_map.at<float>(y, x) = pix;
		}
	}

	return 0;
}


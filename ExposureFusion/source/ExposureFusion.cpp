#include "ExposureFusion.h"


ExposureFusion::ExposureFusion(char* seqPath)  // 构造函数
{
	this->m_state = 0;
	this->m_nframes = 0;

	// Read file list
	string format = std::string(".jpg");  // .jpg .png
	vector<string> img_paths;
	int N_files = this->get_files_format(seqPath, format, img_paths);

	format = std::string(".png");
	N_files = this->get_files_format(seqPath, format, img_paths);
	this->m_nframes = (int)img_paths.size();

	if (this->m_nframes > 0)
	{
		printf("Total %d image files.\n", this->m_nframes);
	}
	else
	{
		printf("[Warning]: non valid images found!\n");
		this->m_state = -1;  // set state
		std::exit(-1);
	}

	// 预先分配内存
	this->m_inputImages.reserve(this->m_nframes);
	this->m_inputGrayImages.reserve(this->m_nframes);

	for (int n = 0; n < m_nframes; n++)
	{
		Mat input_img = cv::imread(img_paths[n], IMREAD_UNCHANGED);

		if (!input_img.data)
		{
			printf("[Err]: Failed to read in image!\n");
			std::exit(-1);
		}

		// Resize for show convenience
		if (input_img.rows > 1000)
		{
			do
			{
				Size sz(int(input_img.cols*0.5), int(input_img.rows*0.5));
				if ((int)(input_img.cols*0.5) % BLOCKCOLS == 0 || (int)(input_img.rows*0.5) % BLOCKROWS == 0)
				{
					sz = Size(int(input_img.cols*0.5 + 1), int(input_img.cols*0.5 + 1));
				}

				// do resizing 
				cv::resize(input_img, input_img, sz, 0.0, 0.0, cv::INTER_CUBIC);
			} while (input_img.rows > 1000);
		}

		Mat gray(input_img.size(), CV_8UC1);
		cv::cvtColor(input_img, gray, CV_BGR2GRAY);
		this->m_inputImages.push_back(input_img);
		this->m_inputGrayImages.push_back(gray);
	}

	std::cout << "finish to read Image Sequence " << endl;
}


const int ExposureFusion::get_files_format(const string & path, const string & format, vector<string>& files)
{
	intptr_t hFile = 0;  // 文件句柄  64位下long 改为 intptr_t
	struct _finddata_t fileinfo;  // 文件信息 
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(), &fileinfo)) != -1)  // 文件存在
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))  // 判断是否为文件夹
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)  // 文件夹名中不含"."和".."
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));  // 保存文件夹名
					this->get_files_format(p.assign(path).append("\\").append(fileinfo.name), format, files);  // 递归遍历文件夹
				}
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));  // 如果不是文件夹，储存文件名
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}

	return int(files.size());
}


void ExposureFusion::QualityMeasuresProcessing()
{
	Mat weightMap;
	Mat contrast;
	Mat saturation;
	Mat wellexposedness;
	Mat originalGray;
	Mat originalColor;

	time_t tok, tic = clock();

	// ---------
	for (int nfrm = 0; nfrm < m_nframes; nfrm++)
	{
		cout << "Quality measure processing - Frame number: " << nfrm + 1;

		QualityMeasures* qm = new QualityMeasures(m_inputImages[nfrm].clone(), m_inputGrayImages[nfrm].clone());

		m_WeightMaps.push_back(qm->getterWeightMap());
		delete(qm);
		qm = nullptr;
	}
	// ---------

	tok = clock();
	cout << endl << "processing time of QualitymeasureProcessing: " << (float)(tok - tic) / CLOCKS_PER_SEC << endl;
	cout << endl;
}

void ExposureFusion::FusionProcessing()
{
	int nframes = getnframes();
	int rows = m_inputImages[0].rows;
	int cols = m_inputImages[0].cols;
	int pyramidDepth = 4;

	setNormalizedWeightMaps();

	vector<Mat> bgr;

	for (int i = 0; i < 3; i++)
	{
		bgr.push_back(setResultByPyramid(i));
	}

	Mat dst;
	cv::merge(bgr, dst);
	/*if(dst.rows > 1000)
		resize(dst, dst, Size(dst.cols*0.5, dst.rows*0.5));*/

	this->m_resultImage = dst.clone();
	/*if (m_resultImage.cols % 4)
		resize(m_resultImage, m_resultImage, Size(m_resultImage.cols - (m_resultImage.cols % 4), m_resultImage.rows));*/
}

void ExposureFusion::setNormalizedWeightMaps()
{
	int nframes = getnframes();
	int rows = m_inputImages[0].rows;
	int cols = m_inputImages[0].cols;

	float sumPix = 0;
#if MODE==GRAY	
	for (int nfrm = 0; nfrm < nframes; nfrm++)
	{
		Mat NorWeightMap(m_inputImages[0].rows, m_inputImages[0].cols, CV_32FC1);

		for (int y = 0; y < rows; y++)
		{
			for (int x = 0; x < cols; x++)
			{
				sumPix = 0;

				for (int n = 0; n < nframes; n++)
				{
					sumPix += m_WeightMaps[n].at<float>(y, x);
				}
				NorWeightMap.at<float>(y, x) = m_WeightMaps[nfrm].at<float>(y, x) / sumPix;
			}
		}

		m_NorWeightMaps.push_back(NorWeightMap);
	}
#endif
}

Mat ExposureFusion::setResultByPyramid(int nch)
{
	int pyramidDepth = 4;
	vector<Mat> gaussianPyramid;
	vector<vector<Mat>> gaussianWeightMapPyramid;
	vector<vector<Mat>> laplacianImagePyramid;
	vector<Mat> fusedPyramid;
	vector<vector<Mat>> fusedPyramidColor;
	const int nframes = getnframes();
	Mat src;
	Mat ucharMap;
	Mat laplimg;
	Mat upGauss;
	Mat cuGauss;
	Mat pvGauss;
	Mat laplacianResult;
	Mat fuseimg;
	Mat rsLaplac;
	Mat result;
	float pix = 0;

#if MODE==GRAY
	vector<Mat> BGR;
	for (int n_frm = 0; n_frm < nframes; n_frm++)
	{
		split(m_inputImages[n_frm], BGR);
		src = BGR[nch].clone();
		laplacianImagePyramid.push_back(vector<Mat>());
		gaussianWeightMapPyramid.push_back(vector<Mat>());
		cv::buildPyramid(src, gaussianPyramid, pyramidDepth);

		ucharMap = Mat(m_NorWeightMaps[n_frm].size(), CV_8UC1);
		for (int y = 0; y < ucharMap.rows; y++)
		{
			for (int x = 0; x < ucharMap.cols; x++)
			{
				pix = m_NorWeightMaps[n_frm].at<float>(y, x) * 255;
				pix = (pix > 255) ? 255 : pix;
				pix = (pix < 0) ? 0 : pix;
				ucharMap.at<uchar>(y, x) = (uchar)pix;
			}
		}

		cv::buildPyramid(ucharMap, gaussianWeightMapPyramid[n_frm], pyramidDepth);

		for (int i = 1; i < gaussianPyramid.size(); i++)
		{
			Mat prev = gaussianPyramid[i - 1].clone();
			Mat crnt = gaussianPyramid[i].clone();

			cv::pyrUp(crnt, crnt, prev.size());
			laplimg = Mat(prev.size(), CV_8SC1);

			for (int y = 0; y < prev.rows; y++)
			{
				for (int x = 0; x < prev.cols; x++)
				{
					laplimg.at<char>(y, x) = prev.at<uchar>(y, x) - crnt.at<uchar>(y, x);
				}
			}

			laplacianImagePyramid[n_frm].push_back(laplimg.clone());
			prev.release();
			crnt.release();
			laplimg.release();
		}

		laplacianImagePyramid[n_frm].push_back(gaussianPyramid[pyramidDepth].clone());
		ucharMap.release();
	}

	cout << "Set laplacian image pyramid " << endl << "Set gaussian weight map pyramid" << endl;
	for (int l = 0; l < pyramidDepth; ++l)
	{
		//cout << "pyramid depth: " << l << endl;
		laplacianResult = Mat(laplacianImagePyramid[0][l].size(), CV_32SC1);
		for (int y = 0; y < laplacianResult.rows; y++)
		{
			for (int x = 0; x < laplacianResult.cols; x++)
			{
				pix = 0;
				for (int nfrm = 0; nfrm < nframes; nfrm++)
				{
					pix += gaussianWeightMapPyramid[nfrm][l].at<uchar>(y, x) * laplacianImagePyramid[nfrm][l].at<char>(y, x);
				}

				laplacianResult.at<int>(y, x) = int(pix / 255.0f);
			}
		}

		fusedPyramid.push_back(laplacianResult.clone());
		laplacianResult.release();
	}


	laplacianResult = Mat(laplacianImagePyramid[0][pyramidDepth].size(), CV_8UC1);
	for (int y = 0; y < laplacianResult.rows; ++y)
	{
		for (int x = 0; x < laplacianResult.cols; ++x)
		{
			pix = 0;
			for (int n_frm = 0; n_frm < nframes; ++n_frm)
			{
				pix += gaussianWeightMapPyramid[n_frm][pyramidDepth].at<uchar>(y, x) * (laplacianImagePyramid[n_frm][pyramidDepth].at<uchar>(y, x));
			}

			laplacianResult.at<uchar>(y, x) = (uchar)(pix / 255);	// uchar char; int
		}
	}
	fusedPyramid.push_back(laplacianResult.clone());

	cout << "Set fused pyramid" << endl;

	int i_pix = 0;
	Mat temp = fusedPyramid[pyramidDepth].clone();
	Mat fusedLaplacianImage = fusedPyramid[pyramidDepth - 1].clone();
	int rows = fusedLaplacianImage.rows;
	int cols = fusedLaplacianImage.cols;

	Mat sumimg(Size(cols, rows), CV_8UC1);
	pyrUp(temp, temp, fusedLaplacianImage.size());

	for (int y = 0; y < temp.rows; y++)
	{
		for (int x = 0; x < temp.cols; x++)
		{
			i_pix = temp.at<uchar>(y, x) + fusedLaplacianImage.at<int>(y, x);
			i_pix = (i_pix > 255) ? 255 : i_pix;
			i_pix = (i_pix < 0) ? 0 : i_pix;
			sumimg.at<uchar>(y, x) = (uchar)i_pix;
		}
	}

	for (int i = pyramidDepth - 2; i >= 0; i--)
	{
		fusedLaplacianImage = fusedPyramid[i].clone();

		pyrUp(sumimg, sumimg, fusedLaplacianImage.size());
		for (int y = 0; y < sumimg.rows; y++)
		{
			for (int x = 0; x < sumimg.cols; x++)
			{
				i_pix = fusedLaplacianImage.at<int>(y, x) + sumimg.at<uchar>(y, x);
				i_pix = (i_pix > 255) ? 255 : i_pix;
				i_pix = (i_pix < 0) ? 0 : i_pix;
				sumimg.at<uchar>(y, x) = (uchar)i_pix;
			}
		}
	}

	result = sumimg.clone();
#endif

	return result;
}

bool ExposureFusion::SaveImageBMP(const char* filename)
{
	if (!strcmp(".bmp", &filename[strlen(filename) - 4]))
	{
		FILE* pFile = NULL;
		fopen_s(&pFile, filename, "wb");
		if (!pFile)
		{
			return false;
		}

		int m_nChannels = m_resultImage.channels();
		int m_nHeight = m_resultImage.rows;
		int m_nWidth = m_resultImage.cols;
		int m_nWStep = (m_nWidth*m_nChannels * sizeof(uchar) + 3)&~3;

		BITMAPFILEHEADER fileHeader;
		fileHeader.bfType = 0x4D42; // 'BM'
		fileHeader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + m_nWStep * m_nHeight + (m_nChannels == 1) * 1024;
		fileHeader.bfReserved1 = 0;
		fileHeader.bfReserved2 = 0;
		fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + (m_nChannels == 1) * 256 * sizeof(RGBQUAD);

		std::fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, pFile);

		BITMAPINFOHEADER infoHeader;
		infoHeader.biSize = sizeof(BITMAPINFOHEADER);
		infoHeader.biWidth = m_nWidth;
		infoHeader.biHeight = m_nHeight;
		infoHeader.biPlanes = 1;
		infoHeader.biBitCount = m_nChannels * 8;
		infoHeader.biCompression = BI_RGB;
		infoHeader.biSizeImage = m_nWStep * m_nHeight;
		infoHeader.biClrImportant = 0;
		infoHeader.biClrUsed = 0;
		infoHeader.biXPelsPerMeter = 0;
		infoHeader.biYPelsPerMeter = 0;

		std::fwrite(&infoHeader, sizeof(BITMAPINFOHEADER), 1, pFile);

		if (m_nChannels == 1)
		{
			for (int l = 0; l < 256; l++)
			{
				RGBQUAD GrayPalette = { byte(l), byte(l), byte(l), 0 };
				std::fwrite(&GrayPalette, sizeof(RGBQUAD), 1, pFile);
			}
		}

		int r;
		for (r = m_nHeight - 1; r >= 0; r--)
		{
			std::fwrite(&m_resultImage.data[r*m_nWStep], sizeof(BYTE), m_nWStep, pFile);
		}

		std::fclose(pFile);
		return true;
	}
	else
	{
		return false;
	}
}
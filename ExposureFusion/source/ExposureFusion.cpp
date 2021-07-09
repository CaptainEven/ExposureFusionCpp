#include "ExposureFusion.h"


// 构造函数
ExposureFusion::ExposureFusion(const char* seq_path, const bool do_resize)
{
	this->m_state = 0;
	this->m_nframes = 0;

	// Read file list
	string format = std::string(".jpg");  // .jpg and .png image
	vector<string> img_paths;
	int N_files = this->getFilesFormat(seq_path, format, img_paths);

	format = std::string(".png");
	N_files = this->getFilesFormat(seq_path, format, img_paths);
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

	for (int fr_i = 0; fr_i < m_nframes; fr_i++)
	{
		Mat img_color = cv::imread(img_paths[fr_i], IMREAD_UNCHANGED);
		if (fr_i == 0)
		{
			this->m_rows = img_color.rows;
			this->m_cols = img_color.cols;

			// Initialize m_imgs_color and m_imgs_gray
			this->m_imgs_color = vector<Mat>(m_nframes, Mat::zeros(m_rows, m_cols, CV_8UC3));
			this->m_imgs_gray = vector<Mat>(m_nframes, Mat::zeros(m_rows, m_cols, CV_8UC1));
		}

		if (!img_color.data)
		{
			printf("[Err]: Failed to read in image!\n");
			std::exit(-1);
		}

		// Resize for show convenience
		if (do_resize)
		{
			if (img_color.rows > 1000)
			{
				do
				{
					Size sz(int(img_color.cols*0.5f), int(img_color.rows*0.5f));
					if ((int)(img_color.cols*0.5f) % BLOCKCOLS == 0 || (int)(img_color.rows*0.5f) % BLOCKROWS == 0)
					{
						sz = Size(int(img_color.cols*0.5 + 1), int(img_color.cols*0.5 + 1));
					}

					// do resizing 
					cv::resize(img_color, img_color, sz, 0.0, 0.0, cv::INTER_CUBIC);

				} while (img_color.rows > 1000);
			}
		}

		// Get gray image
		Mat img_gray(img_color.size(), CV_8UC1);
		cv::cvtColor(img_color, img_gray, CV_BGR2GRAY);

		// Push color and gray image
		this->m_imgs_color[fr_i] = img_color;
		this->m_imgs_gray[fr_i] = img_gray;
	}

	std::cout << "finish to read Image Sequence " << endl;
}



void ExposureFusion::qualityMeasuresProcessing()
{
	// Initialize m_weightMaps
	this->m_weightMaps = vector<Mat>(this->m_nframes, Mat::zeros(m_imgs_color[0].size(), CV_32FC1));

	// Fill the Vector of Mat
	for (int fr_i = 0; fr_i < this->m_nframes; fr_i++)
	{
		QualityMeasures qm = QualityMeasures(this->m_imgs_color[fr_i], this->m_imgs_gray[fr_i]);
		this->m_weightMaps[fr_i] = qm.getWeightMap();  // CV_32FC1
	}
}


void ExposureFusion::fuse()
{
	const int& nframes = getnframes();
	const int& rows = m_imgs_color[0].rows;
	const int& cols = m_imgs_color[0].cols;
	int pyramid_depth = 4;

	this->setNormalizedWeightMaps();

	// Vector initialization
	vector<Mat> bgr(3, cv::Mat::zeros(rows, cols, CV_8U));

	// Fill the vector of Mat
	for (int i = 0; i < 3; i++)
	{
		Mat channel;
		this->setResultByPyramid(i, channel);
		bgr[i] = channel;
	}

	cv::merge(bgr, this->m_resultImage);
}

void ExposureFusion::setNormalizedWeightMaps()
{
	const int& nframes = getnframes();
	const int& rows = m_imgs_color[0].rows;
	const int& cols = m_imgs_color[0].cols;

	float sum_pix = 0.0f;

	// Initialization of the weight map vector
	this->m_normWeightMaps.resize(nframes, Mat::zeros(rows, cols, CV_32F));

	// Fill the vector of Mat
	for (int i = 0; i < nframes; i++)
	{
		Mat norm_weight_map = Mat::zeros(m_imgs_color[0].size(), CV_32FC1);

		for (int y = 0; y < rows; y++)
		{
			for (int x = 0; x < cols; x++)
			{
				sum_pix = 0.0f;

				for (int j = 0; j < nframes; j++)
				{
					sum_pix += this->m_weightMaps[j].at<float>(y, x);
				}

				norm_weight_map.at<float>(y, x) = m_weightMaps[i].at<float>(y, x) / sum_pix;
			}
		}

		this->m_normWeightMaps[i] = norm_weight_map;
	}
}


int ExposureFusion::setResultByPyramid(const int ch, Mat& channel)
{
	const int pyr_max_level = 4;
	const int nframes = getnframes();

	vector<Mat> gauss_pyramid, fused_pyramid;
	vector<vector<Mat>> gauss_weight_map_pyramid;
	vector<vector<Mat>> lap_img_pyramid;
	vector<vector<Mat>> fusedPyramidColor;

	Mat uchar_map, lap_img, lap_result;

	float pix = 0.0f;

	vector<Mat> BGR;
	for (int fr_i = 0; fr_i < nframes; fr_i++)  // process each frame 
	{
		cv::split(this->m_imgs_color[fr_i], BGR);
		const Mat&src = BGR[ch];

		lap_img_pyramid.push_back(vector<Mat>());
		gauss_weight_map_pyramid.push_back(vector<Mat>());
		cv::buildPyramid(src, gauss_pyramid, pyr_max_level);

		uchar_map = Mat::zeros(this->m_normWeightMaps[fr_i].size(), CV_8U);
		for (int y = 0; y < uchar_map.rows; y++)
		{
			for (int x = 0; x < uchar_map.cols; x++)
			{
				pix = this->m_normWeightMaps[fr_i].at<float>(y, x) * 255.0f;
				CLAMP(pix, 0.0f, 255.0f);
				uchar_map.at<uchar>(y, x) = (uchar)pix;
			}
		}

		cv::buildPyramid(uchar_map, gauss_weight_map_pyramid[fr_i], pyr_max_level);

		for (int i = 1; i < gauss_pyramid.size(); i++)  // process each pyramid
		{
			const Mat& pre = gauss_pyramid[i - 1];
			const Mat& cur = gauss_pyramid[i];

			Mat cur_up;
			cv::pyrUp(cur, cur_up, pre.size());

			lap_img = Mat::zeros(pre.size(), CV_8SC1);
			for (int y = 0; y < pre.rows; y++)
			{
				for (int x = 0; x < pre.cols; x++)
				{
					lap_img.at<char>(y, x) = pre.at<uchar>(y, x) - cur_up.at<uchar>(y, x);
				}
			}

			lap_img_pyramid[fr_i].push_back(lap_img.clone());
			lap_img.release();
		}

		lap_img_pyramid[fr_i].push_back(gauss_pyramid[pyr_max_level].clone());
		uchar_map.release();
	}

	cout << "Computing Laplace image pyramid " << endl << "Set gaussian weight map pyramid" << endl;
	for (int l = 0; l < pyr_max_level; ++l)  // process each level
	{
		//cout << "pyramid depth: " << l << endl;
		lap_result = Mat::zeros(lap_img_pyramid[0][l].size(), CV_32SC1);
		for (int y = 0; y < lap_result.rows; y++)
		{
			for (int x = 0; x < lap_result.cols; x++)
			{
				// process each pixel
				pix = 0.0f;  // reset each pixel
				for (int fr_i = 0; fr_i < nframes; fr_i++)
				{
					pix += gauss_weight_map_pyramid[fr_i][l].at<uchar>(y, x) * lap_img_pyramid[fr_i][l].at<char>(y, x);
				}

				lap_result.at<int>(y, x) = int(pix / 255.0f);
			}
		}

		//cout << lap_result.type() << endl;
		fused_pyramid.push_back(lap_result.clone());
		lap_result.release();
	}

	lap_result = Mat(lap_img_pyramid[0][pyr_max_level].size(), CV_8UC1);
	for (int y = 0; y < lap_result.rows; ++y)
	{
		for (int x = 0; x < lap_result.cols; ++x)
		{
			pix = 0;
			for (int n_frm = 0; n_frm < nframes; ++n_frm)
			{
				pix += gauss_weight_map_pyramid[n_frm][pyr_max_level].at<uchar>(y, x) * (lap_img_pyramid[n_frm][pyr_max_level].at<uchar>(y, x));
			}

			lap_result.at<uchar>(y, x) = (uchar)(pix / 255.0f);	// uchar char; int
		}
	}
	fused_pyramid.push_back(lap_result.clone());

	// Set result: fusing
	cout << "Fusing pyramid" << endl;
	int i_pix = 0;
	const Mat& temp = fused_pyramid[pyr_max_level];
	const Mat& fused_lap_img = fused_pyramid[pyr_max_level - 1];
	const int& rows = fused_lap_img.rows;
	const int& cols = fused_lap_img.cols;


	Mat temp_up;
	cv::pyrUp(temp, temp_up, fused_lap_img.size());

	channel = Mat::zeros(Size(cols, rows), CV_8UC1);
	for (int y = 0; y < temp_up.rows; y++)
	{
		for (int x = 0; x < temp_up.cols; x++)
		{
			i_pix = temp_up.at<uchar>(y, x) + fused_lap_img.at<int>(y, x);
			CLAMP(i_pix, 0, 255);
			channel.at<uchar>(y, x) = (uchar)i_pix;
		}
	}

	for (int i = pyr_max_level - 2; i >= 0; i--)
	{
		const Mat& fused_lap_img = fused_pyramid[i];

		cv::pyrUp(channel, channel, fused_lap_img.size());
		for (int y = 0; y < channel.rows; y++)
		{
			for (int x = 0; x < channel.cols; x++)
			{
				i_pix = fused_lap_img.at<int>(y, x) + channel.at<uchar>(y, x);
				i_pix = (i_pix > 255) ? 255 : i_pix;
				i_pix = (i_pix < 0) ? 0 : i_pix;
				channel.at<uchar>(y, x) = (uchar)i_pix;
			}
		}
	}

	return 0;
}


const int ExposureFusion::getFilesFormat(const string & path, const string & format, vector<string>& files)
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
					this->getFilesFormat(p.assign(path).append("\\").append(fileinfo.name), format, files);  // 递归遍历文件夹
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


bool ExposureFusion::saveImageBMP(const char* file_path)
{
	if (!strcmp(".bmp", &file_path[strlen(file_path) - 4]))
	{
		FILE* pFile = NULL;
		pFile = std::fopen(file_path, "wb");  // using std for linux compiling
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
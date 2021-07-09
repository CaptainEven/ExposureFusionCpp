#define _CRT_SECURE_NO_WARNINGS
#define WIN32


#include<thread>
#include"ExposureFusion.h"

using namespace std;
using namespace cv;


#define N_THREADS 4


void splitStr(const string& s, vector<string>& tokens, const char& delim = ' ')
{
	tokens.clear();
	size_t lastPos = s.find_first_not_of(delim, 0);
	size_t pos = s.find(delim, lastPos);
	while (lastPos != string::npos)
	{
		tokens.emplace_back(s.substr(lastPos, pos - lastPos));
		lastPos = s.find_first_not_of(delim, pos);
		pos = s.find(delim, lastPos);
	}
}


int getDirs(const string& path, vector<string>& dirs)
{
	intptr_t hFile = 0;  // 文件句柄  64位下long 改为 intptr_t
	struct _finddata_t fileinfo;  // 文件信息 
	string p;
	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)  // 文件存在
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))  // 判断是否为文件夹
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					dirs.push_back(p.assign(path).append("/").append(fileinfo.name));
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}

	return int(dirs.size());
}


int thread_func(const vector<string>& dirs, const string& res_dir)
{
	if (dirs.size() == 0)
	{
		return -1;
	}

	for (int i = 0; i < int(dirs.size()); i++)
	{
		const string& dir_path = dirs[i];
		ExposureFusion EF(dir_path.c_str(), false);
		if (EF.getState() < 0)
		{
			continue;
		}

		EF.qualityMeasuresProcessing();
		cout << "finish to qualityMeasuresProcessing" << endl;
		EF.fuse();
		cout << "finish to fuse" << endl;
		// ----------

		// Save fused image
		vector<string> tokens;
		splitStr(dir_path, tokens, '/');
		char res_f_path[100];
		const auto& dir_id = tokens[2].c_str();
		sprintf(res_f_path, "%s/EF_%d.jpg", res_dir.c_str(), atoi(dir_id));
		cv::imwrite(res_f_path, EF.getResultImage());

		printf("%s saved.\n", res_f_path);
		printf("End processing seq %d.\n\n", i);
	}

	return 0;
}

// TODO: cmd line tool in Windows and Linux
int main()
{
	const char* seq_path = "./data";
	const char* res_path = "./res";

	// ---------- TODO: convert the for loop to multi-thread task
	// Get sub-dirs
	vector<string> dir_names;
	int ret = getDirs("./data", dir_names);
	const int n_dirs = int(dir_names.size());

	vector<thread> threads(N_THREADS);

	// Split task array for each thread
	const int stride = (int)dir_names.size() / N_THREADS;
	const int n_extra = (int)dir_names.size() % N_THREADS;
	for (int i = 0; i < N_THREADS; ++i)
	{	
		vector<string> dirs;
		if (i == N_THREADS - 1)
		{
			dirs.reserve(stride + n_extra);
		}
		else
		{
			dirs.reserve(stride);
		}
		dirs.insert(dirs.begin(), dir_names.begin() + i*stride, dir_names.begin() + (i+1) * stride);

		threads[i] = thread(thread_func, dirs, res_path);
	}

	for (auto& th : threads)
	{
		th.join();
	}

	//for (int i = 1; i <= n_dirs; i++)
	//{
	//	sprintf(seq_top_path, "%s%d", seq_path, i);
	//	printf("Start processing seq %s...\n", seq_top_path);

	//	time_t tok, tic = clock();

	//	// ----------
	//	ExposureFusion EF(seq_top_path, false);
	//	if (EF.getState() < 0)
	//	{
	//		continue;
	//	}

	//	EF.qualityMeasuresProcessing();
	//	cout << "finish to qualityMeasuresProcessing" << endl;
	//	EF.fuse();
	//	cout << "finish to fuse" << endl;
	//	// ----------

	//	tok = clock();
	//	cout << "total processing time : " 
	//		<< (float)(tok - tic) / CLOCKS_PER_SEC << "s" << endl;

	//	//// show result
	//	//char win_name[60];
	//	//sprintf(win_name, "Exposure Fusion HDR %d", i);
	//	//cv::imshow(win_name, EF.getResultImage());
	//	//cv::waitKey();
	//	//cv::destroyWindow(win_name);

	//	sprintf(res_f_path, "%s\\EF_%d.jpg", res_path, i);
	//	cv::imwrite(res_f_path, EF.getResultImage());

	//	printf("%s saved.\n", res_f_path);
	//	printf("End processing seq %d.\n\n", i);
	//	//system("cls");
	//}

	return 0;
}


//sprintf(res_f_path, "%s\\EF_%d.bmp", res_path, i);
//if (!EF.saveImageBMP(res_f_path))
//{
//	cout << "fail to save result image" << endl;
//	return -1;
//}
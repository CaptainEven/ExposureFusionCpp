#define _CRT_SECURE_NO_WARNINGS


#include<thread>
#include"ExposureFusion.h"

using namespace std;
using namespace cv;


//#define N_THREADS 4  // number of threads


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


int thread_func(const vector<string>& dirs, const string& res_dir, const int& th_id)
{
	if (dirs.size() == 0)
	{
		return -1;
	}

	for (int i = 0; i < int(dirs.size()); i++)
	{
		const string& dir_path = dirs[i];
		cout << "Pocessing " << dir_path << "..." << endl;

		// ----------
		ExposureFusion EF(dir_path.c_str(), false);
		if (EF.getState() < 0)
		{
			continue;
		}

		EF.qualityMeasuresProcessing();
		EF.fuse();
		// ----------

		// Save fused image
		vector<string> tokens;
		splitStr(dir_path, tokens, '/');
		char res_f_path[100];
		const auto& dir_name = tokens[tokens.size() - 1].c_str();
		sprintf(res_f_path, "%s/EF_%s.jpg", res_dir.c_str(), dir_name);
		cv::imwrite(res_f_path, EF.getResultImage());
		//cout << res_f_path << " saved.\n";
		cout << dir_path << " processed in thread#" << th_id << "\n\n";
	}

	return 0;
}

// CMD line tool in Windows and Linux
int main(int argc, char** argv)
{
	if (argc < 4)
	{
		cout << "[Usage]: input_dir(string) output_dir(string) n_threads(int)" << endl;
		exit(-1);
	}
	const char* seq_path = argv[1];
	const char* res_path = argv[2];

	// ---------- multi-thread task
	// Get sub-dirs
	vector<string> dir_names;
	int ret = getDirs(seq_path, dir_names);
	const int n_dirs = int(dir_names.size());

	cout << "\n";
	time_t tok, tic = clock();

	const uint in_nthreads = uint(atoi(argv[3]));
	assert(in_nthreads > 0);

	const uint N_THREADS = MIN(in_nthreads, thread::hardware_concurrency());
	const uint stride = (uint)dir_names.size() / N_THREADS;
	const uint n_extra = (uint)dir_names.size() % N_THREADS;

	vector<thread> threads(N_THREADS);
	for (uint i = 0; i < N_THREADS; ++i)
	{	
		// Split task
		vector<string> thread_dirs;
		if (i == 0)
		{
			thread_dirs.reserve(stride + n_extra);
			thread_dirs.insert(thread_dirs.begin(),
				dir_names.begin() + i * stride,
				dir_names.begin() + (i + 1) * stride + n_extra);
		}
		else
		{
			thread_dirs.reserve(stride);
			thread_dirs.insert(thread_dirs.begin(), 
				dir_names.begin() + i * stride + n_extra,
				dir_names.begin() + (i + 1) * stride + n_extra);
		}

		// Launch threads
		cout << "\nLaunching thread#" << i << " for " << thread_dirs.size() << "sub-dirs.\n";
		threads[i] = thread(thread_func, thread_dirs, res_path, i);
	}

	for (auto& th : threads)
	{
		th.join();
	}

	tok = clock();
	cout << "\nTotal processing time: "
		<< (float)(tok - tic) / CLOCKS_PER_SEC << "s" << endl;

	return 0;
}
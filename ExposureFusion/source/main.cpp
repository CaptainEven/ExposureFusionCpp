#define _CRT_SECURE_NO_WARNINGS

#include "ExposureFusion.h"

using namespace std;
using namespace cv;


// TODO: 做成命令行工具
int main()
{
	const char* seq_path = "./data/";
	char seq_top_path[256];

	const char* res_path = "./res/";
	char res_f_path[256];
	for (int i = 1; i <= 36; i++)
	{
		sprintf(seq_top_path, "%s%d", seq_path, i);
		printf("Start processing seq %s...\n", seq_top_path);

		time_t tok, tic = clock();

		// ----------
		ExposureFusion EF(seq_top_path, false);
		if (EF.getState() < 0)
		{
			continue;
		}

		EF.qualityMeasuresProcessing();
		cout << "finish to qualityMeasuresProcessing" << endl;
		EF.fuse();
		cout << "finish to fuse" << endl;
		// ----------

		tok = clock();
		cout << "total processing time : " 
			<< (float)(tok - tic) / CLOCKS_PER_SEC << "s" << endl;

		//// show result
		//char win_name[60];
		//sprintf(win_name, "Exposure Fusion HDR %d", i);
		//cv::imshow(win_name, EF.getResultImage());
		//cv::waitKey();
		//cv::destroyWindow(win_name);

		sprintf(res_f_path, "%s\\EF_%d.jpg", res_path, i);
		cv::imwrite(res_f_path, EF.getResultImage());

		printf("%s saved.\n", res_f_path);
		printf("End processing seq %d.\n\n", i);
		//system("cls");
	}

	return 0;
}


//sprintf(res_f_path, "%s\\EF_%d.bmp", res_path, i);
//if (!EF.saveImageBMP(res_f_path))
//{
//	cout << "fail to save result image" << endl;
//	return -1;
//}
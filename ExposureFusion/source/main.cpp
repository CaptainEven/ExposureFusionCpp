#define _CRT_SECURE_NO_WARNINGS

#include "ExposureFusion.h"

using namespace std;
using namespace cv;


int main(void)
{
	const char* seq_path = "./data/";
	char seq_top_path[256];

	const char* res_path = "./res/";
	//char res_f_name[256];
	for (int i = 1; i <= 15; i++)
	{
		printf("Start processing seq %d...\n", i);
		sprintf(seq_top_path, "%s%d", seq_path, i);

		time_t tok, tic = clock();

		// ----------
		ExposureFusion EF(seq_top_path);
		if (EF.getState() < 0)
		{
			continue;
		}

		EF.QualityMeasuresProcessing();
		cout << "finish to QualityMeasuresProcessing" << endl;
		EF.FusionProcessing();
		cout << "finish to FusionProcessing" << endl;
		// ----------

		tok = clock();
		cout << endl << "total processing time : " 
			<< (float)(tok - tic) / CLOCKS_PER_SEC << "s" << endl;

		// show result
		char win_name[60];
		sprintf(win_name, "Exposure Fusion HDR %d", i);
		cv::imshow(win_name, EF.getResultImage());
		cv::waitKey();
		cv::destroyWindow(win_name);

		/*sprintf(res_f_name, "%s\\EF_%d.bmp", res_path, i);
		if (!EF.SaveImageBMP(res_f_name))
		{
			cout << "fail to save result image" << endl;
			return -1;
		}*/

		printf("End processing seq %d.\n\n", i);
		//system("cls");
	}

	return 0;
}
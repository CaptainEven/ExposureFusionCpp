#define _CRT_SECURE_NO_WARNINGS

#include "ExposureFusion.h"

using namespace std;
using namespace cv;


int main(void)
{
	const char* seq_path = "./data/";
	char seq_top_path[256];

	const char* res_path = "./res/";
	char res_f_name[256];
	//FILE* fp = fopen("eftime.txt", "at");
	for (int i = 1; i <= 5; i++)
	{
		sprintf(seq_top_path, "%s%d", seq_path, i);

		time_t tok, tic = clock();

		// ----------
		ExposureFusion EF(seq_top_path);
		EF.QualityMeasuresProcessing();
		cout << "finish to QualityMeasuresProcessing" << endl;
		EF.FusionProcessing();
		cout << "finish to FusionProcessing" << endl;
		// ----------

		tok = clock();
		cout << endl << "total processing time : " 
			<< (float)(tok - tic) / CLOCKS_PER_SEC << "s" << endl;
		//fprintf(fp, "%.2f\n", (float)(tok - tic) / CLOCKS_PER_SEC);		

		// show result
		cv::imshow("ExposureFusion HDR", EF.getResultImage());
		cv::waitKey();

		/*sprintf(res_f_name, "%s\\EF_%d.bmp", res_path, i);
		if (!EF.SaveImageBMP(res_f_name))
		{
			cout << "fail to save result image" << endl;
			return -1;
		}*/
		//waitKey();
	}
	//fclose(fp);

	return 0;
}
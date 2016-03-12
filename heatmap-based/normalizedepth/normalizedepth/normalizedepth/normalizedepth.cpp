// normalizedepth.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp> 
#include<opencv2/core/core.hpp>
#define datasetdir "G:\\nyu_hand_dataset_v2\\dataset\\train\\"
#define st 1 
#define en 72756
using namespace std;
using namespace cv;
int minh = 1111, minw = 1111, maxh = 0, maxw = 0;
void worknormalized()
{
	FILE *fout = fopen("avedepth.txt", "w");
	for (int i = st; i <= en; i++)
	{
		char filename[111];
		sprintf(filename, "%s%d%s","J:\\cnnhandtotal\\cnntraindata\\size100\\", i, ".png");
		Mat img = imread(filename);
		resize(img, img, Size(224, 224));
		
		int sum = 0;	
		int cnt = 0;
		for (int h = 0; h < 224; h++)
		{
			for (int w = 0; w < 224; w++)
			{
				if (img.at<uchar>(h, w) < 10) continue;
				cnt++;
				sum = sum + img.at<uchar>(h, w);
			}
		}
		float ave = sum / 255.0 / cnt;
		cout << i << " : " << ave << "\n";
		fprintf(fout, "%.4f\n", ave);
	}
	fclose(fout);

}
void workorigin()
{
	char filenameimg[111];
	char filenameskin[111];
	FILE *fout = fopen("avedepth.txt", "w");
	Mat src;
	for (int i = st; i <= en; i++)
	{
		sprintf(filenameimg, "%sdepth_%d_%07d.png", datasetdir, 1, i);
		Mat img = imread(filenameimg);
		sprintf(filenameskin, "%ssynthdepth_%d_%07d.png", datasetdir, 1, i);
		Mat skin = imread(filenameskin);
		/*imshow("", skin);
		waitKey(0);*/
		minh = 1111;
		minw = 1111;
		maxh = 0;
		maxw = 0;
		for (int h = 1; h < 480; h++)
		{
			for (int w = 1; w < 640; w++)
			{
				if (!(skin.at<Vec3b>(h, w)[0] == 0 && skin.at<Vec3b>(h,w)[1]==0))
				{
					//cout << h << " " << w << "\n";
					minh = min(minh, h);
					maxh = max(maxh, h);
					minw = min(minw, w);
					maxw = max(maxw, w);
				}
			}
		}
		//cout << minh << " " << maxh << " " << minw << " " << maxw << "\n";
		minh = max(0, minh - 10); maxh = min(480, maxh + 10);
		minw = max(0, minw - 10); maxw = min(640, maxw + 10);	
		if (maxw - minw > maxh - minh)
		{
			int delrow = maxh - minh, delcol = maxw - minw;
			int t = min(minh, (delcol - delrow) / 2);
			if (maxh + (delcol - delrow) - t > 480)
				t = maxh - 480 + (delcol - delrow);
			minh = minh - t;
			maxh = maxh + (delcol - delrow) - t;
		}
		else
		{
			int delcol = maxw - minw, delrow = maxh - minh;
			int t = min(minw, (delrow - delcol) / 2);
			if (maxw + (delrow - delcol) - t>640)
				t = maxw - 640 + (delrow - delcol);
			minw = minw - t;
			maxw = maxw + (delrow - delcol) - t;
		}
		//cout << minh << " " << maxh << " " << minw << " " << maxw << "\n";
		int sum = 0;
		int cnt = 0;
		src = Mat::zeros(Size(maxw - minw + 1, maxh - minh + 1), CV_8UC3);
		int h = 0, w = 0;
		for (h = minh; h <= maxh; h++)
		{
			for (w = minw; w <= maxw; w++)
			{
				
				if (skin.at<Vec3b>(h, w)[0] == 0) continue;
				int depth = img.at<Vec3b>(h, w)[1] * 255 + img.at<Vec3b>(h, w)[0];
				//cout << h-minh << " " << w-minw << "\n";
				//src.at<Vec3b>(h - minh, w - minw)[0] = depth % 255;
				sum += depth;
				cnt++;
			}
		}
		//imshow("", src);
		//waitKey(0);
		float ave = (float)sum / cnt;
		cout << i << " : " << ave << "\n";
		fprintf(fout, "%.4f\n", ave);
	}
	fclose(fout);
}
int main()
{
	workorigin();
	//worknormalized();
	return 0;
}

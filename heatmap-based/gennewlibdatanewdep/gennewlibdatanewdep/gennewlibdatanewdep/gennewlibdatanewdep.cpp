// gennewlibdata.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp> 
#include<opencv2/core/core.hpp>

#define synthdir "J:\\cnnhandtotal\\cnntraindata\\size100\\"
#define outputsynthdepdir "J:\\cnnhandtotal\\cnntraindata\\size224\\synthdep\\"
#define outputdepdir "J:\\cnnhandtotal\\cnntraindata\\size224\\dep\\"
#define outputnewdepdir "J:\\cnnhandtotal\\cnntraindata\\size224\\newdep\\"
#define nyudir "G:\\nyu_hand_dataset_v2\\dataset\\train\\"
using namespace std;
using namespace cv;
float dep[72758];
int a[500][500][3];
int minh, maxh, minw, maxw;
float ave0;

Mat changesize(Mat src, int size)
{
	Mat middle = src;
	int del = (int)((1 - size / 100.0) / 2 * src.rows);
	resize(middle, middle, Size(src.cols - 2 * del, src.rows - 2 * del));
	//imshow("", middle);
	//waitKey(0);
	Mat dst = Mat::zeros(Size(src.cols, src.rows), CV_8UC1);
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			dst.at<uchar>(row, col) = 255;

		}
	}

	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			//cout << row << " " << col << "\n";
			if (row >= (int)((1 - size / 100.0) / 2 * src.rows + 0.5) && row < src.rows - del &&
				col >= (int)((1 - size / 100.0) / 2 * src.rows + 0.5) && col < src.cols - del)
			{
				int trow = row - del;
				int tcol = col - del;
				//cout << row << " " << col << " " << trow << " " << tcol << " ";
				//printf("%d\n",middle.at<uchar>(trow, tcol));
				dst.at<uchar>(row, col) = middle.at<uchar>(trow, tcol);

			}
			else
			{
				//dst.at<Vec3b>(row, col) = (255,255,255);
			}
		}
	}
	return dst;
}
void init()
{

}
int main()
{
	init();
	char filenameimg[111];
	char filenameskin[111];
	for (int i = 0; i < 72756; i++)
	{
		sprintf(filenameimg, "%sdepth_%d_%07d.png", nyudir, 1, i + 1);
		Mat img = imread(filenameimg);

		sprintf(filenameskin, "%ssynthdepth_%d_%07d.png", nyudir, 1, i + 1);
		Mat skin = imread(filenameskin);

		minh = 1111;
		minw = 1111;
		maxh = 0;
		maxw = 0;
		for (int h = 1; h < 480; h++)
		{
			for (int w = 1; w < 640; w++)
			{
				if (!(skin.at<Vec3b>(h, w)[0] == 0 && skin.at<Vec3b>(h, w)[1] == 0))
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

		int sum = 0, cnt = 0;
		for (int h = minh; h <= maxh; h++)
		{
			for (int w = minw; w <= maxw; w++)
			{

				if (skin.at<Vec3b>(h, w)[0] == 0) continue;
				int depth = img.at<Vec3b>(h, w)[1] * 255 + img.at<Vec3b>(h, w)[0];
				sum += depth;
				cnt++;
			}
		}
		float ave = sum / cnt;
		if (i == 0) ave0 = ave;

		int mind = 1111, maxd = 0;
		for (int h = minh; h <= maxh; h++)
		{
			for (int w = minw; w <= maxw; w++)
			{

				if (skin.at<Vec3b>(h, w)[0] == 0) continue;
				int depth = img.at<Vec3b>(h, w)[1] * 255 + img.at<Vec3b>(h, w)[0];
				if (depth > 1900) continue;
				mind = min(mind, depth);
				maxd = max(maxd, depth);

			}
		}
		cout << mind << " " << maxd << " " << "\n";
		Mat middle = Mat::zeros(Size(maxw - minw + 1, maxh - minh + 1), CV_8UC1);
		for (int h = minh; h <= maxh; h++)
		{
			for (int w = minw; w <= maxw; w++)
			{
				middle.at<uchar>(h - minh, w - minw) = 255;
				if (skin.at<Vec3b>(h, w)[0] == 0) continue;
				int depth = img.at<Vec3b>(h, w)[1] * 255 + img.at<Vec3b>(h, w)[0];
				//if (depth != 0) cout << h << " " << w << " " << depth << "\n";
				middle.at<uchar>(h - minh, w - minw) = (1.0 - float(depth - mind) / (maxd - mind))*255.0;

			}
		}


		middle = changesize(middle, (float)ave0 / max(ave0, ave)*100.0);
		//first scale according to average depth		
		//imshow("", middle);
		//waitKey(0);
		Mat out = Mat::zeros(Size(224, 224), CV_8UC1);
		resize(middle, out, Size(224, 224));
		char savename[111];
		sprintf(savename, "%s%d%s", outputnewdepdir, i + 1, ".png");
		imwrite(savename, out);
		imshow("", out);
		waitKey(1);
	}

	return 0;
}



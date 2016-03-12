// gentraindata.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp> 
#include<opencv2/core/core.hpp>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<iostream>
#include<algorithm>
#include<highgui.h>
#define st 51000
#define en 72756
#define datasetdir "F:\\cnnhandtotal\\cnntraindata\\size100\\"
#define localizationpicprefix "F:\\cnnhandtotal\\cnntraindata\\loc100"
const int sizemode[3] = { 96, 48, 24 };
using namespace std;
using namespace cv;
double a[111][111];
void localization(int id)
{
	cout << "Local Contrast Normalization Current Id: " << id << "\n";
	char filename[111];
	sprintf(filename, "%s%d%s", datasetdir, id, ".png");

	IplImage* src = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);   //载入灰度图像，深度为8U
	CvSize cvsize;
	cvsize.height = cvsize.width = 96;
	IplImage* middle = cvCreateImage(cvsize, src->depth, CV_LOAD_IMAGE_GRAYSCALE);	
	cvResize(src, middle, 1);
	//IplImage* dst = cvCreateImage(dstcvsize, IPL_DEPTH_64F, 1);
	//IplImage* dst = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_64F, 1);
	IplImage* dst = cvCreateImage(cvSize(middle->width, middle->height), IPL_DEPTH_64F, 1);
	//cvResize(src, dst, 1);
	//cvConvertScale(src, dst, 1.0);
	cvConvertScale(middle, dst, 1.0);
	Mat savez = Mat::zeros(96, 96, CV_64FC1);
	for (int j = 0; j < 96; j++)
	{
		for (int k = 0; k < 96; k++)
		{
			savez.at<double>(j, k) = ((double *)(dst->imageData + dst->widthStep*k))[j];
		}
	}
	IplImage *img[3];
	for (int mode = 0; mode < 3; mode++)
	{
		img[mode] = cvCreateImage(cvSize(sizemode[mode], sizemode[mode]), IPL_DEPTH_8U, 1);
	}
	for (int mode = 0; mode < 3; mode++)
	{
		int size = sizemode[mode];
		Mat z = Mat::zeros(size, size, CV_64FC1);
		resize(savez, z, Size(size, size));
		double aveall = 0;
		for (int row = 0; row < size; row++)
		{
			for (int col = 0; col < size; col++)
			{
				aveall += z.at<double>(row, col);
			}
		}
		aveall /= (size*size);
		double stdall = 0.0;
		for (int row = 0; row < size; row++)
		{
			for (int col = 0; col < size; col++)
			{
				stdall += pow(z.at<double>(row, col) - aveall, 2);
			}
		}
		stdall /= (size*size);
		stdall = sqrt(stdall);
		Mat z2 = z;
		double pmin = 1111.0, pmax = 0.0;
		for (int row = 0; row < size; row++)
		{
			for (int col = 0; col < size; col++)
			{
				double ave = 0;
				int cnt = 0;
				for (int k = max(0, row - (4 - 1 - mode)); k <= min(size - 1, row + (4 - 1 - mode)); k++)
				{
					for (int l = max(0, col - (4 - 1 - mode)); l <= min(size - 1, col + (4 - 1 - mode)); l++)
					{
						ave += z2.at<double>(k, l);
						cnt++;
					}
				}
				ave = ave / cnt;
				double stdnow = 0;
				for (int k = max(0, row - (4 - 1 - mode)); k <= min(size - 1, row + (4 - 1 - mode)); k++)
				{
					for (int l = max(0, col - (4 - 1 - mode)); l <= min(size - 1, col + (4 - 1 - mode)); l++)
					{
						stdnow += pow(z2.at<double>(k, l) - ave, 2);
					}
				}
				stdnow = stdnow / cnt;
				stdnow = sqrt(stdnow);
				a[row][col] = (z.at<double>(row, col) - ave) / (stdnow + stdall);
				pmin = min(pmin, a[row][col]);
				pmax = max(pmax, a[row][col]);
			}
		}
		//IplImage* img = cvCreateImage(cvSize(size, size), IPL_DEPTH_8U, 1);
		for (int row = 0; row < size; row++)
		{
			for (int col = 0; col < size; col++)
			{
				((uchar *)(img[mode]->imageData + img[mode]->widthStep*col))[row] = int((a[row][col] - pmin) / (pmax - pmin)*255.0);				
			}
		}
		char savefile[111];
		sprintf(savefile, "%s%d%s%d%s", localizationpicprefix,  size,"\\",id, ".png");
		cvSaveImage(savefile, img[mode]);
	}
}
void origin()
{
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char synthname[111];
		sprintf(synthname, "%s%07d%s", "I:\\nyu_hand_dataset_v2\\dataset\\train\\synthdepth_1_", i, ".png");
		Mat synth = imread(synthname);
		char realdepname[111];
		sprintf(realdepname, "%s%07d%s", "I:\\nyu_hand_dataset_v2\\dataset\\train\\depth_1_", i, ".png");
		Mat realdep = imread(realdepname);
		int lr = 1111, rr = 0, lc = 1111, rc = 0;
		for (int row = 0; row < 480; row++)
		{
			for (int col = 0; col < 640; col++)
			{
				if (synth.at<Vec3b>(row, col)[0] == 0 && synth.at<Vec3b>(row, col)[1] == 0 && synth.at<Vec3b>(row, col)[2] == 0) continue;
				lr = min(lr, row); rr = max(rr, row); lc = min(lc, col); rc = max(rc, col);
			}
		}
		lr = max(0, lr - 10); rr = min(480, rr + 10); lc = max(0, lc - 10); rc = min(640, rc + 10);
		if (rc - lc>rr - lr)
		{
			int delrow = rr - lr, delcol = rc - lc;
			int t = min(lr, (delcol - delrow) / 2);
			if (rr + (delcol - delrow) - t > 480) t = rr - 480 + (delcol - delrow);
			lr -= t; rr += (delcol - delrow) - t;
		}
		else
		{
			int delcol = rc - lc, delrow = rr - lr;
			int t = min(lc, (delrow - delcol) / 2);
			if (rc + (delrow - delcol) - t > 640) t = rc - 640 + (delrow - delcol);
			lc -= t; rc += (delrow - delcol) - t;
		}
		Mat z = Mat::zeros(Size(rr - lr, rc - lc), CV_8UC3);
		for (int row = lr; row <= rr; row++)
		{
			for (int col = lc; col <= rc; col++)
			{
				if (synth.at<cv::Vec3b>(row, col)[0] == 0 && synth.at<cv::Vec3b>(row, col)[1] == 0 && synth.at<cv::Vec3b>(row, col)[2] == 0) continue;
				z.at<cv::Vec3b>(row - lr, col - lc)[0] = synth.at<cv::Vec3b>(row, col)[0];
				z.at<cv::Vec3b>(row - lr, col - lc)[1] = synth.at<cv::Vec3b>(row, col)[1];
				z.at<cv::Vec3b>(row - lr, col - lc)[2] = synth.at<cv::Vec3b>(row, col)[2];
			}
		}
		//resize(z, z, Size(96, 96));
		//imshow("", z);
		//waitKey(0);
		int mind = 1111, maxd = 0;
		for (int row = 0; row < rr - lr; row++)
		{
			for (int col = 0; col < rc - lc; col++)
			{
				//cout << z.at<cv::Vec3b>(row, col)[0] << " " << z.at<cv::Vec3b>(row, col)[1] << " " << z.at<cv::Vec3b>(row, col)[2] << "\n";
				if (z.at<cv::Vec3b>(row, col)[0] == 0 && z.at<cv::Vec3b>(row, col)[1] == 0 && z.at<cv::Vec3b>(row, col)[2] == 0) continue;
				int dep = z.at<cv::Vec3b>(row, col)[1] * 256 + z.at<cv::Vec3b>(row, col)[0];
				//if (dep < 510) dep = 510; else if (dep>1070) dep = 1070;
				mind = min(mind, dep); maxd = max(maxd, dep);
			}
		}
		Mat dst = Mat::zeros(Size(rr - lr, rc - lc), CV_8UC1);
		for (int row = 0; row < rr - lr; row++)
		{
			for (int col = 0; col < rc - lc; col++)
			{
				if (z.at<cv::Vec3b>(row, col)[0] == 0 && z.at<cv::Vec3b>(row, col)[1] == 0 && z.at<cv::Vec3b>(row, col)[2] == 0)
				{
					dst.at<uchar>(row, col) = 255;
					continue;
				}
				int dep = z.at<cv::Vec3b>(row, col)[1] * 256 + z.at<cv::Vec3b>(row, col)[0];
				dst.at<uchar>(row, col) = int(double(dep - mind) / (maxd - mind)*255.0);
			}
		}
		//imshow("", dst);
		//waitKey(0);
		char filename[111];
		sprintf(filename, "%s%d%s", "J:\\cnnhandtotal\\cnntraindata\\res96\\", i, ".png");
		imwrite(filename, dst);
	}
}
void changesize(int size)
{
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char srcname[111];
		sprintf(srcname,"%s%d%s", "J:\\cnnhandtotal\\cnntraindata\\res96\\", i, ".png");
		IplImage* srcipl = cvLoadImage(srcname, CV_LOAD_IMAGE_GRAYSCALE);   //载入灰度图像，深度为8U
		Mat src = imread(srcname,0);
		Mat middle=src;
		
		int del = (int)((1 - size / 100.0) / 2 * srcipl->height);
		resize(middle, middle, Size(srcipl->height-2*del,srcipl->width-2*del));
		//imshow("", middle);
		//waitKey(0);
		Mat dst = Mat::zeros(Size(srcipl->height, srcipl->width),CV_8UC1);		
		for (int row = 0; row < srcipl->height; row++)
		{
			for (int col = 0; col < srcipl->width; col++)
			{
				//cout << row << " " << col << "\n";
				if (row >= del && row<  srcipl->height -del &&
					col >= del  && col< srcipl->width -del)
				{
					int trow = row - del;
					int tcol = col - del;
					//cout << row << " " << col << " " << trow << " " << tcol << " ";
					//printf("%d\n",middle.at<uchar>(trow, tcol));
					dst.at<uchar>(row, col) = middle.at<uchar>(trow, tcol);
				}
				else
				{
					dst.at<uchar>(row, col) = 255;
				}
			}
		}
		//imshow("", dst);
		//waitKey(0);
		char savename[111];
		sprintf(savename, "%s%d%s%d%s", "J:\\cnnhandtotal\\cnntraindata\\size", size, "\\", i, ".png");
		imwrite(savename, dst);
	}
}
void getall()
{	
	int cnt = 0;
	for (int i = 0; i < 5; i++)
	{
		for (int j = 1; j <= 72756; j++)
		{			
			cout << i << " " << j << "\n";
			char zname[111];
			sprintf(zname, "%s%d%s%d%s", "F:\\cnnhandtotal\\cnntraindata\\size", 100 - i * 5, "\\", j, ".png");
			Mat z = imread(zname);
			char outname[111];
			sprintf(outname, "%s%d%s", "F:\\cnnhandtotal\\cnntraindata\\allpic\\", ++cnt, ".png");
			imwrite(outname, z);
		}
	}
}
void flip()
{
	Mat z[1500];
	for (int i = 50; i <= 700; i++)
	{
		z[i] = Mat::zeros(Size(i, i), CV_8UC1);
	}
	for (int i = 98964; i <= 100000; i++)
	{
		cout << i << "\n";
		char picname[111];
		sprintf(picname, "%s%d%s", "F:\\cnnhandtotal\\cnntraindata\\allpic\\", i, ".png");
		IplImage* srcpic = cvLoadImage(picname, CV_LOAD_IMAGE_GRAYSCALE);   //载入灰度图像，深度为8U
		Mat src = imread(picname,0);
		//Mat z = Mat::zeros(Size(srcpic->height, srcpic->width), CV_8UC1);
		for (int row = 0; row < srcpic->height; row++)
		{
			for (int col = 0; col < srcpic->width; col++)
			{
				z[srcpic->height].at<uchar>(row, srcpic->width -1 - col) = src.at<uchar>(row, col);
			}
		}
		//imshow("", z[srcpic->height]);
		//waitKey(0);
		char filename[111];
		sprintf(filename,"%s%d%s", "F:\\cnnhandtotal\\cnntraindata\\allpic\\", i + 363780, ".png");
		imwrite(filename, z[srcpic->height]);
	}
}
int main()
{
	//origin();
	changesize(85);
	//getall();
	//flip();
	/*for (int id = st; id <= en; id++)
	{
		localization(id);
	}*/
	return 0;
}

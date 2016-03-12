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
#include<ctime>
#include<eigen/dense>
#define originprefix "J:\\cnnhandtotal\\cnntraindata\\size100\\"
#define st 1
#define en 72756
#define NUM 5
using namespace Eigen;
using namespace std;
using namespace cv;
float m[6];
void rotateImage(Mat img, Mat img_rotate, int degree)
{
	
	Point2f center;
	center.x = float(img.cols / 2.0 + 0.5);
	center.y = float(img.rows / 2.0 + 0.5);
	
	Mat M = Mat(2, 3, CV_32F, m);
	M = getRotationMatrix2D(center, degree, 1);
	

	//imshow("", img);
	//waitKey(0);
	//±ä»»Í¼Ïñ£¬²¢ÓÃºÚÉ«Ìî³äÆäÓàÖµ  
	warpAffine(img, img_rotate, M, Size(img.cols,img.rows), 1, BORDER_CONSTANT, Scalar(255, 255, 255));
	//imshow("", img_rotate);
	//waitKey(0);
	//imshow("", img_rotate);
	//waitKey(0);
}
Mat translate(Mat src)
{
	int del = (int)src.rows*0.08;
	int xx = rand() % del - del / 2;
	int yy = rand() % del - del / 2;
	Mat dst = Mat::zeros(Size(src.cols,src.rows),CV_8UC1);
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
			dst.at<uchar>(row, col) = 255;
	}
	
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			if (row + xx >= 0 && row + xx < src.rows && col + yy >= 0 && col + yy < src.cols)
			{
				dst.at<uchar>(row + xx, col + yy) = src.at<uchar>(row, col);

			}
		}
	}
	return dst;
}
Mat flip(Mat src)
{
	int t = rand() % 2;
	if (t == 0) return src;
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col <= src.cols / 2; col++)
		{
			int tmp = src.at<uchar>(row, col);
			src.at<uchar>(row, col) = src.at<uchar>(row, src.cols - 1 - col);
			src.at<uchar>(row, src.cols - 1 - col) = tmp;
		}
	}
	return src;
}
Mat changesize(Mat src,int size)
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
			//cout << row << " " << col << "\n";
			if (row >= del && row<  src.rows - del &&
				col >= del  && col< src.cols - del)
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
	return dst;
}
int main()
{
	srand(time(0));
	for (int i = st; i <= en; i++)
	{
		for (int flipflag = 0; flipflag < 2; flipflag++)
		{
			char srcname[111];
			sprintf(srcname, "%s%d%s", "J:\\cnnhandtotal\\cnntraindata\\allpic\\", i+363780*flipflag, ".png");


			for (int j = 1; j <= NUM; j++)
			{
				Mat src = imread(srcname, 0);

				cout << i << " " << j << "\n";
				if (j > 1)
				{
					//Translation
					//src = translate(src);
					//imshow("1", src);
					//waitKey(0);
					//Rotation
					//Mat mid = src;
					//rotateImage(src, mid, rand() % 30 - 15);
					//imshow("2", src);
					//waitKey(0);
					//Flip Horizontally
					//src = flip(mid);
					//imshow("3", src);
					//waitKey(0);
					src = changesize(src, rand() % 20 + 80);

				}
				char filename[111];
				sprintf(filename, "%s%d%s%d%s", "G:\\spatialcnndata\\", i, "_", j+flipflag*5, ".png");
				imwrite(filename, src);
			}
		}
		//cout << i << " ";
		
		
	}
	return 0;
}
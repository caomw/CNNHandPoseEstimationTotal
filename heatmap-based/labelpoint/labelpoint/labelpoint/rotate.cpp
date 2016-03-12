// labelpoint.cpp : ¶¨Òå¿ØÖÆÌ¨Ó¦ÓÃ³ÌÐòµÄÈë¿Úµã¡£
//

#include "stdafx.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp> 
#include <highgui.h>
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<eigen/dense>
using namespace std;
using namespace cv;
using namespace Eigen;
Mat dst, tmp;
MatrixXd A(2, 2);
void rotateImage(Mat img, Mat img_rotate, int degree)
{
	//Ðý×ªÖÐÐÄÎªÍ¼ÏñÖÐÐÄ  
	Point2f center;
	center.x = float(500 / 2.0 + 0.5);
	center.y = float(500 / 2.0 + 0.5);
	//¼ÆËã¶þÎ¬Ðý×ªµÄ·ÂÉä±ä»»¾ØÕó  
	float m[6];
	Mat M = Mat(2, 3, CV_32F, m);
	M = getRotationMatrix2D(center, degree, 1);
	tmp = Mat::zeros(500, 500, CV_8UC1);
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			m[i * 3 + j] = M.at<double>(i, j);
			if (j<2) A(i, j) = m[i * 3 + j];
		}
	}
	VectorXd h;
	VectorXd x(2);
	x[0] = 298-m[2];
	x[1] = 84-m[5];
	h = (A.inverse())*x;
	cout << h[0] << " " << h[1] << "\n";

	for (int i = 0; i < 500; i++)
	{
		for (int j = 0; j < 500; j++)
		{
			double xx = M.at<double>(0, 0)*i + M.at<double>(0, 1)*j + M.at<double>(0, 2);
			double yy = M.at<double>(1, 0)*i + M.at<double>(1, 1)*j + M.at<double>(1, 2);

			if (xx >= 0 && xx < 500 && yy >= 0 && yy < 500)
			{
				int tx = (int)xx;
				int ty = (int)yy;
				double u = xx - tx;
				double v = yy - ty;
				double res = (1 - u)*(1 - v)*  img.at<uchar>(tx, ty);
				if (ty + 1 < 500) res = res + (1 - u)*v*img.at<uchar>(tx, ty + 1);
				if (tx + 1 < 500) res = res + u*(1 - v)*img.at<uchar>(tx + 1, ty);
				if (tx + 1<500 && ty + 1<500) res = res + u*v*img.at<uchar>(tx + 1, ty + 1);
				tmp.at<uchar>(i, j) = (int)(res + 0.5);
			}
			else

			{
				tmp.at<uchar>(i, j) = 0;
			}
		}
	}
	cout << M.at<double>(1, 1) << "\n";

	//±ä»»Í¼Ïñ£¬²¢ÓÃºÚÉ«Ìî³äÆäÓàÖµ  
	warpAffine(img, img_rotate, M, Size(500, 500), 1, BORDER_CONSTANT, Scalar(0, 0, 0));
}

int main()
{
	Mat src = imread("D:\\CNN\\handjoint\\showjointonpicture\\27.png");
	dst = src;
	imshow("1", src);
	rotateImage(src, dst, -10);
	waitKey(0);
	imshow("1", dst);
	waitKey(0);
	imshow("1", tmp);
	waitKey(0);
	return 0;
}

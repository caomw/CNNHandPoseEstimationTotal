#include "stdafx.h"
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<cmath>
#include<queue>
#include<opencv2/opencv.hpp> 
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <highgui.h>
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main()
{
	
	for (int i = 1; i <= 3423; i++)
	{
		cout << i << "\n";
		char filename[111];
		sprintf(filename, "%s%d%s", "F:\\cnnhandtotal\\rdfcnn\\pic2\\", i, ".png");				
		IplImage* src = cvLoadImage(filename);
		Mat srcimg = imread(filename);		
		Mat z = Mat::zeros(src->width,src->height,CV_8UC1);
		int mind = 11111, maxd = 0;
		for (int j = 0; j < src->height; j++)
		{
			for (int k = 0; k < src->width; k++)
			{
				if (srcimg.at<Vec3b>(j, k)[0]==0 && srcimg.at<Vec3b>(j,k)[1]==0) continue;
				//printf("%d %d %d    ", srcimg.at<Vec3b>(j, k)[0], srcimg.at<Vec3b>(j, k)[1], srcimg.at<Vec3b>(j, k)[2]);
				//cout << srcimg.at<Vec3b>(j, k)[0] << " " << srcimg.at<Vec3b>(j, k)[1] << " " << srcimg.at<Vec3b>(j, k)[2] << "\n";
				int t = srcimg.at<Vec3b>(j,k)[1] * 256 + srcimg.at<Vec3b>(j,k)[0];
				mind = min(mind, t);
				maxd = max(maxd, t);
			}
		}
		for (int j = 0; j < src->height; j++)
		{
			for (int k = 0; k < src->width; k++)
			{
				if (srcimg.at<Vec3b>(j, k)[0] == 0 && srcimg.at<Vec3b>(j, k)[1] == 0)
				{
					z.at<uchar>(j, k) = 255;
					continue;
				}
				int t = srcimg.at<Vec3b>(j,k)[1] * 256 + srcimg.at<Vec3b>(j,k)[0];
				double newd = double(t - mind) / (maxd - mind)*255.0;
				z.at<uchar>(j, k) = (int)newd;
			}
		}
		//cout << mind << " " << maxd << "\n"; 
		resize(z, z, Size(96, 96));
		//imshow("", z);
		//waitKey(0);
		sprintf(filename, "%s%d%s", "F:\\cnnhandtotal\\rdfcnn\\picgray\\", i, ".png");
		imwrite(filename, z);
	}
	return 0;
}
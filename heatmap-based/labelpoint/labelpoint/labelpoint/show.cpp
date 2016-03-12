// labelpoint.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp> 
#include<opencv2/core/core.hpp>
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#define st 1
#define en 2235
#define datasetdir "D:\\CNN\\kinectdata\\0827\\res96\\"
#define jointprefix "D:\\CNN\\kinectdata\\0827\\"
using namespace std;
using namespace cv;


int main1()
{
	for (int i = st; i <= en; i++)
	{
		char filename[111];
		sprintf(filename, "%s%d%s", datasetdir, i, ".png");
		Mat img = imread(filename);	
		resize(img, img, Size(500, 500));
		for (int j = 0; j < 14; j++)
		{			
			sprintf(filename, "%s%d%s%d%s", jointprefix, j, "\\", i, ".txt");
			FILE *fin = fopen(filename, "r");
			int x, y;
			fscanf(fin, "%d %d", &x, &y);
			fclose(fin);			
			circle(img, Point(x,y), 3, Scalar(0, 0, 255), -2);			
		}
		imshow("", img);
		waitKey(0);
	}
	return 0;
}

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
#define en 1672
#define datasetdir "D:\\CNN\\kinectdata\\0824\\res96\\"
#define jointprefix "D:\\CNN\\kinectdata\\0824\\13\\"
using namespace std;
using namespace cv;
int i = 0;
void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	
		
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << i << "\n";
		cout << x << " " << y << "\n";
		char filename[111];
		sprintf(filename, "%s%d%s", jointprefix, i, ".txt");
		FILE *fout = fopen(filename, "w");
		fprintf(fout, "%d %d", x, y);
		fclose(fout);
		i++;
		char s_img[111];
		sprintf(s_img, "%s%d%s", datasetdir, i, ".png");
		Mat img = imread(s_img);
		resize(img, img, Size(500, 500));
		imshow("src", img);
		setMouseCallback("src", on_mouse, 0);
	}

}

int main2()
{
	i = st;
	while (i<=en)
	{
		char s_img[111];
		sprintf(s_img, "%s%d%s", datasetdir, i, ".png");
		Mat img = imread(s_img);
		resize(img, img, Size(500, 500));
		imshow("src", img);
		setMouseCallback("src", on_mouse, 0);
				
		waitKey(0);
	}
	
	return 0;
}

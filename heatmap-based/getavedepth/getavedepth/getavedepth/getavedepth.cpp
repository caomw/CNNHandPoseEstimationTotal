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
#include<sstream>
#include<omp.h>
#define maxz 5
using namespace std;
using namespace cv;
double a[11][11];
int main()
{
	Mat img = imread("J:\\cnnhandtotal\\cnntraindata\\local96\\2.png",0);
	imshow("", img);
	waitKey(0);
	double mind = 1111,maxd = 0;
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			int row = 40 + i * 5, col = 30 + j * 5;
			//from (row,col) to (row+5,col+5)
			double sum = 0;
			for (int k = 0; k < 5; k++)
			{
				for (int l = 0; l < 5; l++)
				{
					sum = sum + img.at<uchar>(row + k, col + j);
				}
			}
			sum = sum / 25.0 / 255.0;
			maxd = max(maxd, sum);
			mind = min(mind, sum);
			a[i][j] = sum;
			cout << i << " " << j << " " << sum << "\n";
		}
	}
	cout << "--------------------------------------------\n";
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			a[i][j] = (a[i][j] - mind) / (maxd - mind)*maxz;
			cout << a[i][j] << " ";
		}
		cout << "\n";
	}
	cout << "--------------------------------------------\n";
	double t = a[4][2];
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			a[i][j] = a[i][j] - t;
			cout << a[i][j] << " ";
		}
		cout <<  "\n";
	}
	return 0;
}


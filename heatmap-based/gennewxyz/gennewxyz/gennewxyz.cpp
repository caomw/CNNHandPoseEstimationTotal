// gennewxyz.cpp : 定义控制台应用程序的入口点。
//

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
#define MAXN 72758
#define S 224
#define Z 2.0
#define NUM 1

using namespace std;
float all[MAXN][93];
float gtz[33];
void work()
{
	FILE *fin = fopen("J:\\cnnhandtotal\\libhandmodelexp\\xyzgt.txt", "r");
	for (int i = 0; i < NUM; i++)
	{
		float ave = 0.0;
		for (int j = 0; j < 31; j++)
		{
			fscanf(fin, "%f", &gtz[j]);
			ave += gtz[j];
		}
		ave /= 31.0;
		float std = 0.0;
		for (int j = 0; j < 31; j++)
		{
			std += pow(gtz[j] - ave, 2);
		}
		std = sqrt(std);
		for (int j = 0; j < 31; j++)
		{
			gtz[j] = 2.0 + (gtz[j] - ave) / std;
			//printf("%.4f\n",gtz[j]);
		}
	}
	fclose(fin);
}
int main()
{
	work();
	
	FILE *fin = fopen("J:\\cnnhandtotal\\libhandmodelexp\\xyz224.txt", "r");
	FILE *fout = fopen("D:\\CNN\\handmodel\\libmodel\\initialnew.txt", "w");
	for (int i = 0; i < NUM; i++)
	{
		for (int j = 0; j < 93; j++)
			fscanf(fin, "%f", &all[i][j]);
		for (int j = 0; j < 31; j++)
		{
			float z = gtz[j];
			float x = (all[i][j * 3] * 100.0 / S - 0.5)*z;
			float y = (0.5-all[i][j * 3+1] * 100.0 / S)*z;
			
			//cout << x << " " << y << " "<<z << "\n";
			if (i==NUM-1) fprintf(fout, "%.2f %.2f %.2f\n", x, y, z);
		}
	}
	fclose(fin);
	fclose(fout);
	return 0;
}
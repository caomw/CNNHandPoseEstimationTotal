#include"stdafx.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp> 
#include<opencv2/core/core.hpp>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<iostream>
#include<eigen/dense>  
#define eps 1e-6
#define oridir "D:\\CNN\\lstmheatmapalljoint\\lstmheatmapalljoint\\lstmheatmapalljoint\\target.txt"
#define st 1
#define en 8248
#define N 484
#define Height 22
#define Width 22
#define showdir "J:\\cnnhandtotal\\cnntraindata\\res96\\"
#define S 500
using namespace std;
using namespace cv;
using namespace Eigen;
double z[23][23];
VectorXd dB(N);
MatrixXd A(N, 4);
double xx0, yy0, a1 = 1.0, sigma = 1.0;
double resx[111111][14], resy[111111][14];
double calc(int row, int col, double x0, double y0, double a1, double sigma)
{
	return a1*exp(-pow(col - x0 - 0.5, 2) / (2 * sigma*sigma) - pow(row - y0 - 0.5, 2) / (2 * sigma*sigma));
}
void init(double src[Height + 1][Width + 1])
{

	int pos = 0;
	for (int i = 1; i <= Width; i++)
	for (int j = 1; j <= Height; j++)
	{
		A(pos, 0) = 1.0*calc(j, i, xx0, yy0, a1, sigma) / a1;																	//df/dA1
		A(pos, 1) = 1.0*calc(j, i, xx0, yy0, a1, sigma)*(i - xx0 - 0.5) / pow(sigma, 2);									//df/dx0
		A(pos, 2) = 1.0*calc(j, i, xx0, yy0, a1, sigma)*(j - yy0 - 0.5) / pow(sigma, 2);									//df/dy0
		A(pos, 3) = 1.0*calc(j, i, xx0, yy0, a1, sigma)*(pow(i - xx0 - 0.5, 2) + pow(j - yy0 - 0.5, 2)) / pow(sigma, 3);    //df/dsigma
		dB(pos) = src[j][i] - calc(j, i, xx0, yy0, a1, sigma);
		++pos;

	}
}
void getcenterpoint()
{
	VectorXd dlambda;
	dlambda = (A.transpose()*A).inverse()*(A.transpose()*dB);

	a1 = a1 + dlambda[0];
	xx0 = xx0 + dlambda[1];
	yy0 = yy0 + dlambda[2];
	sigma = sigma + dlambda[3];
}
void ori()
{
	FILE *fin = fopen(oridir, "r");
	for (int i = st; i <= en; i++)
	{

		for (int joint = 0; joint < 14; joint++)
		{
			double minz = 111, maxz = 0.0;
			for (int j = 1; j <= 22; j++)
			{
				for (int k = 1; k <= 22; k++)
				{
					fscanf(fin, "%lf", &z[j][k]);
					minz = min(minz, z[j][k]);
					maxz = max(maxz, z[j][k]);
					if (abs(z[j][k] - maxz) < eps)
					{
						xx0 = k;
						yy0 = j;
					}
				}
			}
			a1 = 1.0; sigma = 1.0;

			for (int j = 1; j <= 22; j++)
			{
				for (int k = 1; k <= 22; k++)
				{
					z[j][k] = (z[j][k] - minz) / (maxz - minz);
				}
			}
			init(z);
			getcenterpoint();
			resx[i][joint] = xx0; resy[i][joint] = yy0;
			if (abs(maxz) < eps) { resx[i][joint] = resx[i - 1][joint]; resy[i][joint] = resy[i - 1][joint]; }
		}

		
	}
}
void showjointonpicture()
{
	for (int i = st; i <= en; i++)
	{
		char s_img[111];
		cout << "Current Image Id:" << i << "\n";
		sprintf(s_img, "%s%d%s", showdir, i, ".png");
		Mat img = imread(s_img);
		Mat dist;
		resize(img, img, Size(S, S));
		for (int j = 0; j < 14; j++)
		{
			double y = resy[i][j], x = resx[i][j];
			
			if (j == 6 || j == 7 || j == 3) circle(img, Point(y / 22.0*S, x / 22.0*S), 3, Scalar(0, 0, 255), -2);
			if (j == 8 || j == 9 || j == 4) circle(img, Point(y / 22.0*S, x / 22.0*S), 3, Scalar(0, 255, 0), -2);
			if (j == 10 || j == 11 || j == 5) circle(img, Point(y / 22.0*S, x / 22.0*S), 3, Scalar(255, 0, 0), -2);
			if (j == 1) circle(img, Point(y / 22.0*S, x / 22.0*S), 3, Scalar(0, 0, 0), -2);
			if (j == 2 || j == 0) circle(img, Point(y / 22.0*S, x / 22.0*S), 3, Scalar(255, 255, 255), -2);
			if (j == 12 || j == 13) circle(img, Point(y / 22.0*S, x / 22.0*S), 3, Scalar(255, 0, 255), -2);
		}		
		imshow("", img);
		waitKey(1);
		char savename[111];
		sprintf(savename, "%s%d%s", "J:\\cnnhandtotal\\lstmheatmap\\alljoint\\target\\", i, ".png");
		imwrite(savename, img);
	}
}
int main()
{
	ori();
	FILE *fout = fopen("D:\\CNN\\lstmheatmapalljoint\\lstmheatmapalljoint\\lstmheatmapalljoint\\targetout.txt", "w");
	for (int i = st; i <= en; i++)
	{
		for (int j = 0; j < 14; j++)
		{
			fprintf(fout, "%.2f %.2f\n", resx[i][j], resy[i][j]);
		}		
	}
	fclose(fout);
	showjointonpicture();
}

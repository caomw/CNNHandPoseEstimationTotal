#include "stdafx.h"
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<cmath>
#include<queue>
#include <eigen/dense>  
#include<opencv2/opencv.hpp> 
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<sstream>
#include<omp.h>
#define Width 22
#define Height 22
#define N 484
#define MaxLen 111
#define st 0
#define en 251
#define eps 1e-6
#define NUM 10
#define JOINT 14
#define MINLIGHT 0.5
#define CONFIDENCE 5
#define THUMBSQUARE 0.70
#define THUMBMINLIGHT 0.30
#define MIDDLESQUARE 0.80
#define MIDDLEMINLIGHT  0.30
#define UPSQUARE 0.80
#define UPMINLIGHT 0.30
#define NPIC 252
#define S 500
using namespace Eigen;
using namespace std;
using namespace cv;
VectorXd dB(N);
MatrixXd A(N, 4);
char *picdir;
double xx0, yy0, a1 = 1.0, sigma = 1.0,z[Height+1][Width+1], savez[Height+1][Width+1],value[JOINT+1][NUM+1];
double rsquare, lastrsquare,peakrsquare[JOINT+1][NUM+1],peakx0[JOINT+1][NUM+1],peaky0[JOINT+1][NUM+1];
int used[Height+1][Width+1],maxf[JOINT+1][CONFIDENCE+1],*tmpchoose;
int *choose,*tchoose;
int maxid[JOINT + 1][CONFIDENCE + 1];
//Thumb
double minfarthumb[CONFIDENCE + 1], maxclosethumb[CONFIDENCE + 1];  
//Middle
double minfarmiddle[CONFIDENCE + 1], maxclosemiddle[CONFIDENCE + 1], believemiddle[CONFIDENCE + 1][5];
//Up
double minfarup[CONFIDENCE + 1], maxcloseup[CONFIDENCE + 1], believeup[CONFIDENCE + 1][5];
struct pixeltype
{
	int x, y;
	double data;
}pixel[(Height+1)*(Width+1)+11];
struct QMAX
{
	int x, y;
	double data;	
	friend bool operator < (const QMAX &t1, const QMAX &t2)
	{
		return t2.data - t1.data>eps;
	}
	QMAX(int a, int b, double c) :x(a), y(b), data(c){}
	QMAX(){}
};
struct QMIN
{
	int x, y;
	double data;
	friend bool operator < (const QMIN &t1, const QMIN &t2)
	{
		return t1.data - t2.data>eps;
	}
	QMIN(int a, int b, double c) :x(a), y(b), data(c){}
	QMIN(){}
};
priority_queue<QMAX> q1;
priority_queue<QMIN> q2;
bool cmp(pixeltype t1, pixeltype t2)
{
	return t1.data - t2.data>eps;
}

double calc(int row, int col, double x0, double y0, double a1, double sigma)
{
	return a1*exp(-pow(col - x0 - 0.5, 2) / (2 * sigma*sigma) - pow(row - y0 - 0.5, 2) / (2 * sigma*sigma));
}

double computedis2(double xx1, double yy1,double xx2, double yy2)
{
	return sqrt(pow(xx1 - xx2, 2) + pow(yy1 - yy2, 2)) / 22.0*96.0;
}

double computeabs(double xx1, double yy1,double xx2,  double yy2)
{
	return abs(xx1 - xx2) / 22.0*96.0;
}

int checkempty(Mat img, int r1, int r2, int c1, int c2)
{
	int cnt = 0;
	for (int k = r1; k <= r2; k++)
	{
		for (int l = c1; l <= c2; l++)
		{
			if (img.at<cv::Vec3b>(k, l)[0] != 255) cnt++;
		}
	}
	return cnt;
}

void init(double src[Height+1][Width+1])
{
		
	int pos = 0;
	for (int i = 1; i <= Width;i++)
		for (int j = 1;j<= Height;j++)		
		{			
			A(pos, 0) = 1.0*calc(j, i, xx0, yy0, a1, sigma)/a1;																	//df/dA1
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

void computefitness()
{
	double sse = 0.0, sum = 0.0, ave = 0.0, sst = 0.0;
	for (int i = 1; i <= Height; i++)
	{
		for (int j = 1; j <= Width; j++)
		{
			sse = sse + pow(calc(i, j, xx0, yy0, a1, sigma) - z[i][j], 2);
			sum = sum + z[i][j];
		}
	}
	ave = sum / (Height*Width);
	for (int i = 1; i <= Height; i++)
	{
		for (int j = 1; j <= Width; j++)
		{
			sst = sst + pow(z[i][j] - ave, 2);
		}
	}
	rsquare = 1 - sse / sst;
}

int *modifythumb(int choose[JOINT + 1], double value[JOINT + 1][NUM + 1], double peakrsquare[JOINT + 1][NUM + 1], Mat pic, double peakx0[JOINT + 1][NUM + 1], double peaky0[JOINT + 1][NUM + 1], int thumbleftarea, int thumbrightarea, int tooclose)
{
	int flag[CONFIDENCE + 1]; 
	for (int j = 1; j <= CONFIDENCE; j++)
	{
		minfarthumb[j] = 1111; maxclosethumb[j] = 0; flag[j] = 0;
	}
	tmpchoose = (int *)malloc(sizeof(int)*(JOINT + 1));
	for (int j = 1; j <= NUM; j++) //Finger 3
	{
		if (peakrsquare[3 + 1][j] < 0) continue;		
		if (computedis2(peakx0[3 + 1][j], peaky0[3 + 1][j], peakx0[1 + 1][choose[1 + 1]], peaky0[1 + 1][choose[1 + 1]])<14.0 || computedis2(peakx0[3 + 1][j], peaky0[3 + 1][j], peakx0[2 + 1][ choose[2 + 1]], peaky0[2 + 1][choose[2 + 1]])<14.0) continue;
		for (int k = 1; k <= NUM; k++) //Finger 4
		{
			if (peakrsquare[4 + 1][k] < 0) continue;
			if (computedis2(peakx0[4 + 1][k], peaky0[4 + 1][k], peakx0[1 + 1][choose[1 + 1]], peaky0[1 + 1][choose[1 + 1]])<14.0 || computedis2(peakx0[4 + 1][k], peaky0[4 + 1][k], peakx0[2 + 1][choose[2 + 1]], peaky0[2 + 1][choose[2 + 1]])<14.0) continue;
			for (int l = 1; l <= NUM; l++) //Finger 5
			{
				if (peakrsquare[5 + 1][l] < 0) continue;
				if (computedis2(peakx0[5 + 1][l], peaky0[5 + 1][l], peakx0[1 + 1][choose[1 + 1]], peaky0[1 + 1][choose[1 + 1]])<14.0 || computedis2(peakx0[5 + 1][l], peaky0[5 + 1][l], peakx0[2 + 1][choose[2 + 1]], peaky0[2 + 1][choose[2 + 1]])<14.0) continue;
				if (tooclose == 1)
				{
					int mark3 = 0;
					for (int u = 6; u <= 13; u++)
					{
						if (computedis2(peakx0[u + 1][choose[u + 1]], peaky0[u + 1][choose[u + 1]], peakx0[3 + 1][j], peaky0[3 + 1][j]) < 12.0)
						{
							mark3 = 1;
							break;
						}
					}
					int mark4 = 0;
					for (int u = 6; u <= 13; u++)
					{
						if (computedis2(peakx0[u + 1][choose[u + 1]], peaky0[u + 1][choose[u + 1]], peakx0[4 + 1][k], peaky0[4 + 1][k]) < 12.0)
						{
							mark4 = 1;
							break;
						}
					}
					int mark5 = 0;
					for (int u = 6; u <= 13; u++)
					{
						if (computedis2(peakx0[u + 1][choose[u + 1]], peaky0[u + 1][choose[u + 1]], peakx0[5 + 1][l], peaky0[5 + 1][l]) < 12.0)
						{
							mark5 = 1;
							break;
						}
					}
					if (mark3 + mark4 + mark5 >= 2) continue;					
				}
				if (computedis2(peakx0[5 + 1][l], peaky0[5 + 1][l], peakx0[1 + 1][choose[1 + 1]], peaky0[1 + 1][choose[1 + 1]])<14.0 || computedis2(peakx0[5 + 1][l], peaky0[5 + 1][l], peakx0[2 + 1][choose[2 + 1]], peaky0[2 + 1][choose[2 + 1]])<14.0) continue;
				if ((checkempty(pic, max(1, int(peaky0[3 + 1][j] / 22.0*96.0) - 5), min(96, int(peaky0[3 + 1][j] / 22.0*96.0) + 5), max(1, int(peakx0[3 + 1][j] / 22.0*96.0) - 5), min(96, int(peakx0[3 + 1][j] / 22.0*96.0) + 5)) <= 20) + 
					(checkempty(pic, max(1, int(peaky0[4 + 1][k] / 22.0*96.0) - 5), min(96, int(peaky0[4 + 1][k] / 22.0*96.0) + 5), max(1, int(peakx0[4 + 1][k] / 22.0*96.0) - 5), min(96, int(peakx0[4 + 1][k] / 22.0*96.0) + 5)) <= 20) + 
					(checkempty(pic, max(1, int(peaky0[5 + 1][l] / 22.0*96.0) - 5), min(96, int(peaky0[5 + 1][l] / 22.0*96.0) + 5), max(1, int(peakx0[5 + 1][l] / 22.0*96.0) - 5), min(96, int(peakx0[5 + 1][l] / 22.0*96.0) + 5)) <= 20) >= 2) continue;
				double closethumb = min(computedis2(peakx0[3 + 1][j], peaky0[3 + 1][j], peakx0[4 + 1][k], peaky0[4 + 1][k]), (computedis2(peakx0[4 + 1][k], peaky0[4 + 1][k], peakx0[5 + 1][l], peaky0[5 + 1][l])));
				//max(min)
				double farthumb = max(computedis2(peakx0[3 + 1][j], peaky0[3 + 1][j], peakx0[4 + 1][k], peaky0[4 + 1][k]), (computedis2(peakx0[4 + 1][k], peaky0[4 + 1][k], peakx0[5 + 1][l], peaky0[5 + 1][l])));
				//min(max)
				if ((thumbleftarea == 1 && (peakx0[3 + 1][j]<peakx0[0 + 1][choose[0 + 1]]) + (peakx0[4 + 1][k]<peakx0[0 + 1][choose[0 + 1]]) + (peakx0[5 + 1][l]<peakx0[0 + 1][choose[0 + 1]]) == 3)
					|| (thumbrightarea == 1 && (peakx0[3 + 1][j]>peakx0[0 + 1][choose[0 + 1]]) + (peakx0[4 + 1][k]>peakx0[0 + 1][choose[0 + 1]]) + (peakx0[5 + 1][l]>peakx0[0 + 1][choose[0 + 1]]) == 3) 
					|| (thumbleftarea == 0 && thumbrightarea == 0 && (((peakx0[3 + 1][j]<peakx0[0 + 1][choose[0 + 1]]) + (peakx0[4 + 1][k]<peakx0[0 + 1][choose[0 + 1]]) + (peakx0[5 + 1][l]<peakx0[0 + 1][choose[0 + 1]]) == 3)
					|| (peakx0[3 + 1][j]>peakx0[0 + 1][choose[0 + 1]]) + (peakx0[4 + 1][k]>peakx0[0 + 1][choose[0 + 1]]) + (peakx0[5 + 1][l]>peakx0[0 + 1][choose[0 + 1]]) == 3)))
				{

					if ((peakx0[4 + 1][k]>peakx0[3 + 1][j] && peakx0[5 + 1][l]>peakx0[4 + 1][k]) || (peakx0[4 + 1][k]<peakx0[3 + 1][j] && peakx0[5 + 1][l]<peakx0[4 + 1][k]))
					{
						for (int confidence = 1; confidence <= CONFIDENCE; confidence++)
						{
							if ((peakrsquare[3 + 1][j] >= 0.95 - 0.05*confidence) + (peakrsquare[4 + 1][k] >= 0.95 - 0.05*confidence) + (peakrsquare[5 + 1][l] >= 0.95 - 0.05*confidence) >= 3)
							{
								if (closethumb <= 14.0 && farthumb <= 14.0 && farthumb<minfarthumb[confidence] || (farthumb == minfarthumb[confidence] && closethumb>maxclosethumb[confidence]))
								{
									flag[confidence] = 1;
									minfarthumb[confidence] = min(minfarthumb[confidence], farthumb);
									maxclosethumb[confidence] = max(maxclosethumb[confidence], closethumb);
									maxid[3][confidence] = j; maxid[4][confidence] = k; maxid[5][confidence] = l;
								}
							}
						}
					}
				}
			}
		}
	}
	tmpchoose = (int *)malloc(sizeof(int)*(JOINT + 1));
	for (int i = 1; i <= JOINT; i++) tmpchoose[i] = choose[i];
	for (int j = 1; j <= CONFIDENCE; j++)
	{
		if (flag[j])
		{	
			tmpchoose[3 + 1] = maxid[3][j]; tmpchoose[4 + 1] = maxid[4][j]; tmpchoose[5 + 1] = maxid[5][j];
			break;
		}
	}
	return tmpchoose;
}

int *modifymiddle(int choose[JOINT + 1], double value[JOINT + 1][NUM + 1], double peakrsquare[JOINT + 1][NUM + 1], Mat pic, double peakx0[JOINT + 1][NUM + 1], double peaky0[JOINT + 1][NUM + 1], int thumbleftarea, int thumbrightarea)
{
	int flag[CONFIDENCE + 1];
	for (int j = 1; j <= CONFIDENCE; j++)
	{
		minfarmiddle[j] = 1111; maxclosemiddle[j] = 0; flag[j] = 0;
	}
	tmpchoose = (int *)malloc(sizeof(int)*(JOINT + 1));
	for (int j = 1; j <= NUM; j++) //Finger 7
	{
		if (peakrsquare[7 + 1][j]<0 || (peakrsquare[7 + 1][j]>MIDDLESQUARE && value[7 + 1][j] < MIDDLEMINLIGHT)) continue;
		for (int k = 1; k <= NUM; k++) //Finger 9
		{
			if (peakrsquare[9 + 1][k]<0 || (peakrsquare[9 + 1][k]>MIDDLESQUARE && value[9 + 1][k] < MIDDLEMINLIGHT)) continue;
			for (int l = 1; l <= NUM; l++) //Finger 11
			{
				if (peakrsquare[11 + 1][l]<0 || (peakrsquare[11 + 1][l]>MIDDLESQUARE && value[11 + 1][l] < MIDDLEMINLIGHT)) continue;
				for (int m = 1; m <= NUM; m++) //Finger 13
				{
					if (peakrsquare[13 + 1][m]<0 || (peakrsquare[13 + 1][m]>MIDDLESQUARE && value[13 + 1][m] < MIDDLEMINLIGHT)) continue;
					if ((checkempty(pic, max(1, int(peaky0[7 + 1][j] / 22.0*96.0) - 5), min(96, int(peaky0[7 + 1][j] / 22.0*96.0) + 5), max(1, int(peakx0[7 + 1][j] / 22.0*96.0) - 5), min(96, int(peakx0[7 + 1][j] / 22.0*96.0) + 5))<10) +
						(checkempty(pic, max(1, int(peaky0[9 + 1][k] / 22.0*96.0) - 5), min(96, int(peaky0[9 + 1][k] / 22.0*96.0) + 5), max(1, int(peakx0[9 + 1][k] / 22.0*96.0) - 5), min(96, int(peakx0[9 + 1][k] / 22.0*96.0) + 5))<10) +
						(checkempty(pic, max(1, int(peaky0[11 + 1][l] / 22.0*96.0) - 5), min(96, int(peaky0[11 + 1][l] / 22.0*96.0) + 5), max(1, int(peakx0[11 + 1][l] / 22.0*96.0) - 5), min(96, int(peakx0[11 + 1][l] / 22.0*96.0) + 5))<10) +
						(checkempty(pic, max(1, int(peaky0[13 + 1][m] / 22.0*96.0) - 5), min(96, int(peaky0[13 + 1][m] / 22.0*96.0) + 5), max(1, int(peakx0[13 + 1][m] / 22.0*96.0) - 5), min(96, int(peakx0[13 + 1][m] / 22.0*96.0) + 5))<10) >= 2) continue;
					double closemiddle = min(computedis2(peakx0[7 + 1][j], peaky0[7 + 1][j], peakx0[9 + 1][k], peaky0[9 + 1][k]), min(computedis2(peakx0[9 + 1][k], peaky0[9 + 1][k], peakx0[11 + 1][l], peaky0[11 + 1][l]), computedis2(peakx0[11 + 1][l], peaky0[11 + 1][l], peakx0[13 + 1][m], peaky0[13 + 1][m])));
					double closemiddleabs = min(computeabs(peakx0[7 + 1][j], peaky0[7 + 1][j], peakx0[9 + 1][k], peaky0[9 + 1][k]), min(computeabs(peakx0[9 + 1][k], peaky0[9 + 1][k], peakx0[11 + 1][l], peaky0[11 + 1][l]), computeabs(peakx0[11 + 1][l], peaky0[11 + 1][l], peakx0[13 + 1][m], peaky0[13 + 1][m])));
					double farmiddle = max(computedis2(peakx0[7 + 1][j], peaky0[7 + 1][j], peakx0[9 + 1][k], peaky0[9 + 1][k]), max(computedis2(peakx0[9 + 1][k], peaky0[9 + 1][k], peakx0[11 + 1][l], peaky0[11 + 1][l]), computedis2(peakx0[11 + 1][l], peaky0[11 + 1][l], peakx0[13 + 1][m], peaky0[13 + 1][m])));
					double farmiddleabs = max(computeabs(peakx0[7 + 1][j], peaky0[7 + 1][j], peakx0[9 + 1][k], peaky0[9 + 1][k]), max(computeabs(peakx0[9 + 1][k], peaky0[9 + 1][k], peakx0[11 + 1][l], peaky0[11 + 1][l]), computeabs(peakx0[11 + 1][l], peaky0[11 + 1][l], peakx0[13 + 1][m], peaky0[13 + 1][m])));
					if (thumbleftarea == 1)
					{
						if (!(peakx0[13 + 1][m]<peakx0[11 + 1][l] && peakx0[11 + 1][l]<peakx0[9 + 1][k] && peakx0[9 + 1][k]<peakx0[7 + 1][j]))  continue;
					}
					if (thumbrightarea == 1)
					{
						if (!(peakx0[13 + 1][m]>peakx0[11 + 1][l] && peakx0[11 + 1][l]>peakx0[9 + 1][k] && peakx0[9 + 1][k]>peakx0[7 + 1][j])) continue;
					}
					if ((peakx0[9 + 1][k]<peakx0[7 + 1][j] && peakx0[11 + 1][l]<peakx0[9 + 1][k] && peakx0[13 + 1][m]<peakx0[11 + 1][l]) ||
						(peakx0[9 + 1][k]>peakx0[7 + 1][j] && peakx0[11 + 1][l]>peakx0[9 + 1][k] && peakx0[13 + 1][m]>peakx0[11 + 1][l]))
					{
						for (int confidence = 1; confidence <= CONFIDENCE; confidence++)
						{
							if ((peakrsquare[7 + 1][j] >= 0.95 - 0.05*confidence) + (peakrsquare[9 + 1][k] >= 0.95 - 0.05*confidence) +
								(peakrsquare[11 + 1][l] >= 0.95 - 0.05*confidence) + (peakrsquare[13 + 1][m] >= 0.95 - 0.05*confidence) >= 4)
							{
								if (closemiddleabs >= 7.0 &&  farmiddleabs <= 18.0 && farmiddle<24.0 &&
									(closemiddle>maxclosemiddle[confidence] || farmiddle<minfarmiddle[confidence]))
									//(maxclosemiddle[confidence] == 0 || (abs(closemiddle - maxclosemiddle[confidence]) <= 7.0 && abs(farmiddle - minfarmiddle[confidence]) <= 9.0 &&
									//(peakrsquare[7 + 1][j]>believemiddle[confidence][1] || peakrsquare[9 + 1][k]>believemiddle[confidence][2] || peakrsquare[11 + 1][l]>believemiddle[confidence][3] || peakrsquare[13 + 1][m]>believemiddle[confidence][4]))))
									
								{
									flag[confidence] = 1;
									minfarmiddle[confidence] = min(minfarmiddle[confidence], farmiddle);
									maxclosemiddle[confidence] = max(maxclosemiddle[confidence], closemiddle);
									believemiddle[confidence][1] = max(believemiddle[confidence][1], peakrsquare[7 + 1][j]);
									believemiddle[confidence][2] = max(believemiddle[confidence][2], peakrsquare[9 + 1][k]);
									believemiddle[confidence][3] = max(believemiddle[confidence][3], peakrsquare[11 + 1][l]);
									believemiddle[confidence][4] = max(believemiddle[confidence][4], peakrsquare[13 + 1][m]);
									maxid[7][confidence] = j; maxid[9][confidence] = k; maxid[11][confidence] = l;  maxid[13][confidence] = m;
								}
							}
						}
					}
				}
			}
		}
	}
	tmpchoose = (int *)malloc(sizeof(int)*(JOINT + 1));
	for (int i = 1; i <= JOINT; i++) tmpchoose[i] = choose[i];
	for (int j = 1; j <= CONFIDENCE; j++)
	{
		if (flag[j])
		{
			tmpchoose[7 + 1] = maxid[7][j]; tmpchoose[9 + 1] = maxid[9][j]; tmpchoose[11 + 1] = maxid[11][j]; tmpchoose[13 + 1] = maxid[13][j];
			break;
		}
	}
	return tmpchoose;
}

int *modifyup(int choose[JOINT + 1], double value[JOINT + 1][NUM + 1], double peakrsquare[JOINT + 1][NUM + 1], Mat pic, double peakx0[JOINT + 1][NUM + 1], double peaky0[JOINT + 1][NUM + 1], int thumbleftarea, int thumbrightarea)
{
	int flag[CONFIDENCE + 1];
	for (int j = 1; j <= CONFIDENCE; j++)
	{
		minfarup[j] = 1111; maxcloseup[j] = 0; flag[j] = 0;
	}
	for (int j = 1; j <= NUM; j++) //Finger 6
	{
		if (peakrsquare[6 + 1][j]<0 || (peakrsquare[6 + 1][j]>UPSQUARE && value[6 + 1][j] < UPMINLIGHT)) continue;
		for (int k = 1; k <= NUM; k++) //Finger 8
		{
			if (peakrsquare[8 + 1][k]<0 || (peakrsquare[8 + 1][k]>UPSQUARE && value[8 + 1][k] < UPMINLIGHT)) continue;
			{
				for (int l = 1; l <= NUM; l++) //Finger 10
				{
					if (peakrsquare[10 + 1][l]<0 || (peakrsquare[10 + 1][l]>UPSQUARE && value[10 + 1][l] < UPMINLIGHT)) continue;
					for (int m = 1; m <= NUM; m++) //Finger 12
					{
						if (peakrsquare[12 + 1][m]<0 || (peakrsquare[12 + 1][m]>UPSQUARE && value[12 + 1][m] < UPMINLIGHT)) continue;
						double closeup = min(computedis2(peakx0[6 + 1][j], peaky0[6 + 1][j], peakx0[8 + 1][k], peaky0[8 + 1][k]), min(computedis2(peakx0[8 + 1][k], peaky0[8 + 1][k], peakx0[10 + 1][l], peaky0[10 + 1][l]), computedis2(peakx0[10 + 1][l], peaky0[10 + 1][l], peakx0[12 + 1][m], peaky0[12 + 1][m])));
						double closeupabs = min(computeabs(peakx0[6 + 1][j], peaky0[6 + 1][j], peakx0[8 + 1][k], peaky0[8 + 1][k]), min(computeabs(peakx0[8 + 1][k], peaky0[8 + 1][k], peakx0[10 + 1][l], peaky0[10 + 1][l]), computeabs(peakx0[10 + 1][l], peaky0[10 + 1][l], peakx0[12 + 1][m], peaky0[12 + 1][m])));						
						double farup = max(computedis2(peakx0[6 + 1][j], peaky0[6 + 1][j], peakx0[8 + 1][k], peaky0[8 + 1][k]), max(computedis2(peakx0[8 + 1][k], peaky0[8 + 1][k], peakx0[10 + 1][l], peaky0[10 + 1][l]), computedis2(peakx0[10 + 1][l], peaky0[10 + 1][l], peakx0[12 + 1][m], peaky0[12 + 1][m])));
						double farupabs = max(computeabs(peakx0[6 + 1][j], peaky0[6 + 1][j], peakx0[8 + 1][k], peaky0[8 + 1][k]), max(computeabs(peakx0[8 + 1][k], peaky0[8 + 1][k], peakx0[10 + 1][l], peaky0[10 + 1][l]), computeabs(peakx0[10 + 1][l], peaky0[10 + 1][l], peakx0[12 + 1][m], peaky0[12 + 1][m])));
						if (thumbleftarea == 1)
						{
							if (!(peakx0[12 + 1][m]<peakx0[10 + 1][l] && peakx0[10 + 1][l]<peakx0[8 + 1][k] && peakx0[8 + 1][k]<peakx0[6 + 1][j])) continue;
						}
						if (thumbrightarea == 1)
						{
							if (!(peakx0[12 + 1][m]>peakx0[10 + 1][l] && peakx0[10 + 1][l]>peakx0[8 + 1][k] && peakx0[8 + 1][k]>peakx0[6 + 1][j]))  continue;
						}
						if ((peakx0[8 + 1][k]>peakx0[6 + 1][j] && peakx0[10 + 1][l]>peakx0[8 + 1][k] && peakx0[12 + 1][m]>peakx0[10 + 1][l]) ||
							(peakx0[8 + 1][k]<peakx0[6 + 1][j] && peakx0[10 + 1][l]<peakx0[8 + 1][k] && peakx0[12 + 1][m]<peakx0[10 + 1][l]))
						{
							for (int confidence = 1; confidence <= CONFIDENCE; confidence++)
							{
								if ((peakrsquare[6 + 1][j] >= 0.95 - 0.05*confidence) + (peakrsquare[8 + 1][k] >= 0.95 - 0.05*confidence) + (peakrsquare[10 + 1][l] >= 0.95 - 0.05*confidence) + (peakrsquare[12 + 1][m] >= 0.95 - 0.05*confidence) >= 4)
								{
									if (closeupabs >= 8.0 && farup <= 32.0 && farupabs <= 32.0 && 
										(closeup>maxcloseup[confidence] || farup<minfarup[confidence]))
										//(maxcloseup[confidence] == 0 || (abs(closeup - maxcloseup[confidence]) <= 8.0 && abs(farup - minfarup[confidence]) <= 10.0 &&
										//(peakrsquare[6 + 1][j]>believeup[confidence][1] || peakrsquare[8 + 1][k]>believeup[confidence][2] || peakrsquare[10 + 1][l]>believeup[confidence][3] || peakrsquare[12 + 1][m]>believeup[confidence][4]))))
									{
										flag[confidence] = 1;
										minfarup[confidence] = min(minfarup[confidence], farup);
										maxcloseup[confidence] = max(maxcloseup[confidence], closeup);
										believeup[confidence][1] = max(believeup[confidence][1], peakrsquare[6 + 1][j]);
										believeup[confidence][2] = max(believeup[confidence][2], peakrsquare[8 + 1][k]);
										believeup[confidence][3] = max(believeup[confidence][3], peakrsquare[10 + 1][l]);
										believeup[confidence][4] = max(believeup[confidence][4], peakrsquare[12 + 1][m]);
										maxid[6][confidence] = j; maxid[8][confidence] = k; maxid[10][confidence] = l;  maxid[12][confidence] = m;
									}
								}
							}
						}

					}
				}
			}
		}
	}
	tmpchoose = (int *)malloc(sizeof(int)*(JOINT + 1));
	for (int i = 1; i <= JOINT; i++) tmpchoose[i] = choose[i];
	for (int j = 1; j <= CONFIDENCE; j++)
	{
		if (flag[j])
		{
			tmpchoose[6 + 1] = maxid[6][j]; tmpchoose[8 + 1] = maxid[8][j]; tmpchoose[10 + 1] = maxid[10][j]; tmpchoose[12 + 1] = maxid[12][j];
			break;
		}
	}
	return tmpchoose;
}
void showjointonpicture()
{
	for (int i = st+1; i <= en+1; i++)
	{
		//stringstream s_img;
		char s_img[111];
		sprintf(s_img, "%s%d%s", "D:\\CNN\\oridataours\\", i, ".png");
		//s_img << "D:\\CNN\\oridata\\" << i << ".png";
		string s = "D:\\CNN\\joint\\";
		Mat img = imread(s_img);
		Mat dist;

		resize(img, img, Size(S, S));
#pragma omp parallel for
		for (int j = 0; j<14; j++)
		{
			//if (j != 3 && j !=4  && j != 5 && j!=0)  continue;
			//if (j != 6 && j != 8 && j != 10 && j != 12 && j != 7 && j != 9 && j != 11 && j != 13) continue;
			//if (j != 6 && j != 8 && j != 10 && j != 12) continue;
			//if (j != 7 && j != 9 && j != 11 && j != 13) continue;
			//stringstream s_joint;
			char s_joint[111];
			sprintf(s_joint, "%s%d%s%d%s", "D:\\CNN\\c++gaussianfit\\joint\\", i - 1, "_", j, ".txt");
			//s_joint << s << i - 1 << "_" << j << ".txt";
			FILE *fp = fopen(s_joint, "r");
			float x, y;
			fscanf(fp, "%f %f", &y, &x);

			if (j == 6 || j == 7 || j == 3) circle(img, Point(y / 22.0*S, x / 22.0*S), 3, Scalar(0, 0, 255), -2);
			if (j == 8 || j == 9 || j == 4) circle(img, Point(y / 22.0*S, x / 22.0*S), 3, Scalar(0, 255, 0), -2);
			if (j == 10 || j == 11 || j == 5) circle(img, Point(y / 22.0*S, x / 22.0*S), 3, Scalar(255, 0, 0), -2);
			if (j == 1) circle(img, Point(y / 22.0*S, x / 22.0*S), 3, Scalar(0, 0, 0), -2);
			if (j == 2 || j == 0) circle(img, Point(y / 22.0*S, x / 22.0*S), 3, Scalar(255, 255, 255), -2);
			if (j == 12 || j == 13) circle(img, Point(y / 22.0*S, x / 22.0*S), 3, Scalar(255, 0, 255), -2);
			fclose(fp);
		}

		cerr << "i = " << i << endl;
		// imshow("", img);
		//waitKey(0);
		//stringstream s_out;
		//string ss = "D:\\CNN\\showonpicture\\";
		//s_out << ss << i << ".png";
		char s_out[111];
		sprintf(s_out, "%s%d%s", "D:\\CNN\\c++gaussianfit\\showonpicture\\", i, ".png");
		imwrite(s_out, img);
		//waitKey(0);
	}
}
int main()
{
	picdir = "D:\\CNN\\0728\\CNN1\\0728\\res96\\";
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char fitname[111];
		sprintf(fitname, "%s%d%s", "D:\\CNN\\c++gaussianfit\\fit", i, ".txt");
		FILE *foutfit = fopen(fitname, "w");
		choose = (int *)malloc(sizeof(int)*(JOINT + 1));
		tchoose = (int *)malloc(sizeof(int)*(JOINT + 1));
		for (int joint = 0; joint <= 13; joint++)
		{
			char heatmapname[MaxLen];
			sprintf(heatmapname,"%s%d_%d%s", "D:\\CNN\\CNN1\\results\\r_", i, joint, ".png");
			Mat img = imread(heatmapname);					
			int pmin = 11111, pmax = 0;
			//Find Maximum & Minimum
			fprintf(foutfit, "%d : \n", joint);
			for (int k = 0; k < Height; k++)
			{
				for (int l = 0; l < Width; l++)
				{					
					pmax = max(pmax, (int)img.at<cv::Vec3b>(k, l)[0]);
					pmin = min(pmin, (int)img.at<cv::Vec3b>(k, l)[0]);
				}
			}						
			int id=0;
			for (int k = 1; k <= Height; k++)
			{
				for (int l = 1; l <= Width; l++)
				{
					z[k][l] =(double) (img.at<cv::Vec3b>(k-1, l-1)[0] - pmin) / (double)(pmax - pmin);										
					savez[k][l] = z[k][l];
					pixel[id].data = z[k][l]; pixel[id].x = l; pixel[id++].y = k;
				}											
			}			
			//Sort by pixel intensity
			sort(pixel, pixel + Height*Width, cmp);
			int maxrow[NUM + 1], maxcol[NUM + 1];
			memset(used, 0, sizeof(used));
			for (int j = 1; j <= NUM; j++) maxrow[j] = maxcol[j] = -1;
			int num = 1;
			//Top Num Peak
			for (int j = 0; j < Height*Width && num<=NUM;j++)
			{
				if (used[pixel[j].y][pixel[j].x]) continue;
				maxrow[num] = pixel[j].y; maxcol[num] = pixel[j].x;				
				value[joint + 1][num] = pixel[j].data;
				for (int irow = max(1, maxrow[num] - 1); irow <= min(3 - (maxrow[num] - max(1, maxrow[num] - 1) + 1) + maxrow[num], 22); irow++)
				{
					for (int icol = max(1, maxcol[num] - 1); icol <= min(3 - (maxcol[num] - max(1, maxcol[num] - 1) + 1) + maxcol[num], 22); icol++)
					{
						used[irow][icol] = 1;
					}
				}
				num++;
			}
			int T = 2;
			for (num = 1; num <= NUM; num++)
			{
				//fprintf(foutfit, "        %d.   \n", num);
				if (maxrow[num] == -1) continue;
				int row1 = max(1, maxrow[num] - T), row2 = min(2 * T + 1 - (maxrow[num] - max(1, maxrow[num] - T) + 1) + maxrow[num], 22);
				int col1 = max(1, maxcol[num] - T), col2 = min(2 * T + 1 - (maxcol[num] - max(1, maxcol[num] - T) + 1) + maxcol[num], 22);
				double fmax = 0.0;
				pixeltype tmp[26];
				int id = 26;
				tmp[0].data = 0.0;
				q1.empty();
				q2.empty();
				for (int irow = 1; irow <= Height; irow++)
				{
					for (int icol = 1; icol <= Width; icol++)
					{
						if (irow >= row1 && irow <= row2 && icol >= col1 && icol <= col2)
						{
							z[irow][icol] = savez[irow][icol];
							q1.push(QMAX(icol, irow, z[irow][icol]));							
							q2.push(QMIN(icol, irow, z[irow][icol]));
							
						}
						else z[irow][icol] = 0.0;
						
					}
				}								
				if (joint == 2 && num == 7)
				{
					//cout << "----";
				}
				double maxrsquare = -11111.0;
				double maxx0, maxy0;
				xx0 = maxcol[num]; yy0 = maxrow[num];
				a1 = 1.0; sigma = 1.0; lastrsquare = 0.0;
				for (int iter = 1; iter <= 30; iter++)
				{
					init(z);
					getcenterpoint();
					computefitness();
					if (rsquare - lastrsquare < 1e-3) break;
					lastrsquare = rsquare;
				}
				if (rsquare - maxrsquare>eps) { maxrsquare = rsquare; maxx0 = xx0; maxy0 = yy0; }
				//fprintf(foutfit, "    %4d %4d %4d %12.6f %12.6f %12.6f %12.6f %12.6f\n", joint, num, 0, maxrsquare, xx0, yy0, a1, sigma);				
				if (!(maxrsquare - 0.80 > eps))
				{
					for (int j = 1; j <= 13; j++)
					{
						for (int t = 0; t <= 1; t++)
						{
							if (t == 0)
							{
								QMAX now = q1.top();
								q1.pop();
								xx0 = now.x; yy0 = now.y;
							}//forward
							else
							{
								QMIN now = q2.top();
								q1.pop();
								xx0 = now.x; yy0 = now.y;
							}//backward
							a1 = 1.0; sigma = 1.0; lastrsquare = 0.0;
							for (int iter = 1; iter <= 30; iter++)
							{
								init(z);
								getcenterpoint();
								computefitness();
								if (rsquare - lastrsquare < 1e-3) break;
								lastrsquare = rsquare;
							}
							if (rsquare - maxrsquare>eps) { maxrsquare = rsquare; maxx0 = xx0; maxy0 = yy0; }
							//fprintf(foutfit, "    %4d %4d %4d %12.6f %12.6f %12.6f %12.6f %12.6f\n", joint, num,(t==0?j:26-j), rsquare, xx0, yy0, a1, sigma);
						}
						if (maxrsquare - 0.80 > eps)
						{
							break;
						}
					}					
				}				
				//fprintf(foutfit, "\n\n");
				peakrsquare[joint + 1][num] = maxrsquare; peakx0[joint + 1][num] = maxx0; peaky0[joint + 1][num] = maxy0;				
			}		
			for (num = 1; num <= NUM; num++)
			{
				fprintf(foutfit, "    %d.%12.6f %12.6f %12.6f %12.6f\n", num, value[joint + 1][num], peakx0[joint + 1][num], peaky0[joint+ 1][num],peakrsquare[joint + 1][num]);
			}
			double fminrsquare = 11111.0;
			for (num = 1; num <= NUM; num++)
			{
				if (1.0 - peakrsquare[joint + 1][num]<fminrsquare && value[joint + 1][num]>MINLIGHT)
				{
					fminrsquare = 1.0 - peakrsquare[joint + 1][num];
					tchoose[joint+1]=choose[joint + 1] = num;
				}
			}	
			fprintf(foutfit, "\n");
		}
		fclose(foutfit);
		char oripic[111];
		sprintf(oripic, "%s%d%s", picdir, i + 1, ".png");
		Mat pic = imread(oripic);
		resize(pic, pic, Size(96, 96));
		//thumb first
		choose = modifythumb(choose, value, peakrsquare, pic, peakx0, peaky0, 0, 0, 0);
		//middle second
		choose = modifymiddle(choose, value, peakrsquare, pic, peakx0, peaky0, 0, 0);
		//up third
		choose = modifyup(choose, value, peakrsquare, pic, peakx0, peaky0, 0, 0);
		/////Finger 3 4 5 should be at the left of palm
		int thumbleftarea = 0,thumbrightarea = 0;
		if (peakx0[6 + 1][choose[6 + 1]]>peakx0[8 + 1][choose[8 + 1]] && peakx0[8 + 1][choose[8 + 1]]>peakx0[10 + 1][choose[10 + 1]] && peakx0[10 + 1][choose[10 + 1]]>peakx0[12 + 1][choose[12 + 1]] &&
			peakx0[7 + 1][choose[7 + 1]]>peakx0[9 + 1][choose[9 + 1]] && peakx0[9 + 1][choose[9 + 1]]>peakx0[11 + 1][choose[11 + 1]] && peakx0[11 + 1][choose[11 + 1]]>peakx0[13 + 1][choose[13 + 1]] &&
			peakx0[3 + 1][choose[3 + 1]]>peakx0[0 + 1][choose[0 + 1]] && peakx0[4 + 1][choose[4 + 1]]>peakx0[0 + 1][choose[0 + 1]] && peakx0[5 + 1][choose[5 + 1]]>peakx0[0 + 1][choose[0 + 1]])
			thumbleftarea = 1;
		/////Finger 3 4 5 should be at the right of palm
		if (peakx0[6 + 1][choose[6 + 1]]<peakx0[8 + 1][choose[8 + 1]] && peakx0[8 + 1][choose[8 + 1]]<peakx0[10 + 1][choose[10 + 1]] && peakx0[10 + 1][choose[10 + 1]]<peakx0[12 + 1][choose[12 + 1]] && 
			peakx0[7 + 1][choose[7 + 1]]<peakx0[9 + 1][choose[9 + 1]] && peakx0[9 + 1][choose[9 + 1]]<peakx0[11 + 1][choose[11 + 1]] && peakx0[11 + 1][choose[11 + 1]]<peakx0[13 + 1][choose[13 + 1]] && 
			peakx0[3 + 1][choose[3 + 1]]<peakx0[0 + 1][choose[0 + 1]] && peakx0[4 + 1][choose[4 + 1]]<peakx0[0 + 1][choose[0 + 1]] && peakx0[5 + 1][choose[5 + 1]]<peakx0[0 + 1][choose[0 + 1]])
			thumbrightarea = 1;

		if (thumbleftarea == 0 && thumbrightarea == 0)
		{
			if (peakx0[3 + 1][choose[3 + 1]]<peakx0[0 + 1][choose[0 + 1]] && peakx0[4 + 1][choose[4 + 1]]<peakx0[0 + 1][choose[0 + 1]] && peakx0[5 + 1][choose[5 + 1]]<peakx0[0 + 1][choose[0 + 1]])
				thumbleftarea = 1;
			else
			{
				if (peakx0[3 + 1][choose[3 + 1]]>peakx0[0 + 1][choose[0 + 1]] && peakx0[4 + 1][choose[4 + 1]]>peakx0[0 + 1][choose[0 + 1]] && peakx0[5 + 1][choose[5 + 1]]>peakx0[0 + 1][choose[0 + 1]])
					thumbrightarea = 1;
			}
		}
		/////Modify thumb middle up
		if (thumbleftarea!= 0 || thumbrightarea!= 0)
		{
			choose = modifythumb(choose, value, peakrsquare, pic, peakx0, peaky0, thumbleftarea, thumbrightarea, 0);
			choose = modifymiddle(choose, value, peakrsquare, pic, peakx0, peaky0, thumbleftarea, thumbrightarea);
			choose = modifyup(choose, value, peakrsquare, pic, peakx0, peaky0, thumbleftarea, thumbrightarea);
		}
		int tooclose = 0;
		for (int j = 6; j <= 13; j++)
		{
			tooclose = 1;
			break;
		}
		/////Finger 3 4 5 should be modified;
		if (tooclose == 1) choose = modifythumb(choose, value, peakrsquare, pic, peakx0, peaky0, thumbleftarea, thumbrightarea, 1);
		/////Check sucess or not
		tooclose = 0;
		for (int j = 6; j <= 13; j++)
		{
			if (computedis2(peakx0[j + 1][choose[j + 1]], peaky0[j + 1][choose[j + 1]], peakx0[3 + 1][choose[3 + 1]], peaky0[3 + 1][choose[3 + 1]])<5.0)
			{
				tooclose = 1;
				break;
			}
		}
		/////Modify 3 4 5 failure
		if (tooclose == 1)
		{
			choose[3 + 1] = tchoose[3 + 1]; choose[4 + 1] = tchoose[4 + 1]; choose[5 + 1] = tchoose[5 + 1];
		}


		for (int j = 0; j <= 13; j++)
		{
			char jointfile[111];
			sprintf(jointfile, "%s%d_%d%s", "D:\\CNN\\c++gaussianfit\\joint\\", i, j, ".txt");
			FILE *foutjoint = fopen(jointfile, "w");
			fprintf(foutjoint, "%.6f %.6f\n", peakx0[j + 1][choose[j + 1]], peaky0[j + 1][choose[j + 1]]);
			fclose(foutjoint);
		}
	}
	showjointonpicture();
	return 0;
}
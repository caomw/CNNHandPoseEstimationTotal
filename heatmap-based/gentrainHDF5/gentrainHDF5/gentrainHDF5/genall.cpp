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
#include<hdf5.h>
#include<eigen/dense>
#define datasetdir "F:\\cnnhandtotal\\cnntraindata\\allpic\\"
#define jointprefix "F:\\cnnhandtotal\\cnntraindata\\alljointtogether.txt"
#define trainprefix "F:\\cnnhandtotal\\cnntraindata\\local"
int st, en;
const int sizemode[3] = { 96, 48, 24 };
#define file96prefix "J:\\newtrain\\trial96_"
#define file48prefix "J:\\newtrain\\trial48_"
#define file24prefix "J:\\newtrain\\trial24_"
char file96[111], file48[111], file24[111];
#define DIML1 12126
#define DIML2 6776
using namespace std;
using namespace cv;
using namespace Eigen;
int nowcnt = 0;
double ox[14], oy[14];
double rotx[14], roty[14];
double a[111][111], data96[DIML1][1][96][96], data48[DIML1][1][48][48], data24[DIML1][1][24][24], label[DIML1][DIML2];
Mat srcimg;
IplImage *img[3];

void localization()
{
	for (int i = 0; i < 3; i++)
		img[i] = cvCreateImage(cvSize(sizemode[i], sizemode[i]), IPL_DEPTH_8U, 1);
	
	FILE *fjoint = fopen(jointprefix, "r");
	for (int i = 1; i < st; i++)
	{
		double x;
		for (int j = 1; j <= 28; j++) fscanf(fjoint, "%lf", &x);
	}

	

	Mat z[3];
	Mat z2;
	for (int mode = 0; mode < 3; mode++)
	{
		z[mode] = Mat::zeros(sizemode[mode], sizemode[mode], CV_8UC1);
	}
	IplImage* src = NULL;
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char filename[111];
		sprintf(filename, "%s%d%s", datasetdir, i, ".png");		
		for (int j = 0; j < 14; j++)
		{
			fscanf(fjoint, "%lf %lf", &ox[j], &oy[j]);		
		}		
		//srcimg = imread(filename, 0);				
		Mat savez = imread(filename, 0);
		nowcnt++;			
		for (int k = 0; k < 14; k++)
		{
			for (int u = 1; u <= 22; u++)
			{
				for (int v = 1; v <= 22; v++)
				{
					label[nowcnt - 1][k * 484 + u * 22 + v] = exp(-1.0 / (2 * 1 * 1)*(pow(u - ox[k] - 0.5, 2) + pow(v - oy[k] - 0.5, 2)));
				}
			}
		}
		
		
		for (int mode = 0; mode < 3; mode++)
		{
			int size = sizemode[mode];
			
			resize(savez, z[mode], Size(size, size));
			
			for (int row = 0; row < size; row++)
			{
				for (int col = row; col < size; col++)
				{
					int t = z[mode].at<uchar>(row, col);
					z[mode].at<uchar>(row, col) = z[mode].at<uchar>(col, row);
					z[mode].at<uchar>(col, row) = t;
				}
			}
			//imshow("", z[mode]);
			//waitKey(0);
			double aveall = 0;
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					aveall += z[mode].at<uchar>(row, col);
				}
			}
			aveall /= (size*size);
			double stdall = 0.0;
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					stdall += pow(z[mode].at<uchar>(row, col) - aveall, 2);
				}
			}
			stdall /= (size*size);
			stdall = sqrt(stdall);
			z2 = z[mode];
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
							ave += z2.at<uchar>(k, l);
							cnt++;
						}
					}
					ave = ave / cnt;
					double stdnow = 0;
					for (int k = max(0, row - (4 - 1 - mode)); k <= min(size - 1, row + (4 - 1 - mode)); k++)
					{
						for (int l = max(0, col - (4 - 1 - mode)); l <= min(size - 1, col + (4 - 1 - mode)); l++)
						{
							stdnow += pow(z2.at<uchar>(k, l) - ave, 2);
						}
					}
					stdnow = stdnow / cnt;
					stdnow = sqrt(stdnow);
					a[row][col] = (z[mode].at<uchar>(row, col) - ave) / (stdnow + stdall);
					pmin = min(pmin, a[row][col]);
					pmax = max(pmax, a[row][col]);
				}
			}

			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					((uchar *)(img[mode]->imageData + img[mode]->widthStep*row))[col] = int((a[row][col] - pmin) / (pmax - pmin)*255.0);
					switch (mode)
					{
					case 0:data96[nowcnt - 1][0][row][col] = (a[row][col] - pmin) / (pmax - pmin); break;
					case 1:data48[nowcnt - 1][0][row][col] = (a[row][col] - pmin) / (pmax - pmin); break;
					case 2:data24[nowcnt - 1][0][row][col] = (a[row][col] - pmin) / (pmax - pmin); break;
					}
				}
			}
			
			waitKey(0);
			char savefile[111];
			sprintf(savefile, "%s%d%s%d%s", trainprefix, sizemode[mode], "\\", nowcnt, ".png");
			cvSaveImage(savefile, img[mode]);
		}		
	}
	fclose(fjoint);
}
void genHDF5()
{
	//96*96
	hid_t fileid, datasetid, dataspaceid, labelid, labelspaceid;
	herr_t status;
	fileid = H5Fcreate(file96, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dimsdata[4], dimslabel[2];
	dimsdata[0] = DIML1;
	dimsdata[1] = 1;
	dimsdata[2] = 96;
	dimsdata[3] = 96;
	dimslabel[0] = DIML1;
	dimslabel[1] = 6776;
	dataspaceid = H5Screate_simple(4, dimsdata, NULL);
	datasetid = H5Dcreate(fileid, "/data96", H5T_IEEE_F64LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/label96", H5T_IEEE_F64LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(datasetid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data96);
	H5Dwrite(labelid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);
	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	status = H5Dclose(labelid);
	status = H5Sclose(labelspaceid);
	status = H5Fclose(fileid);

	//48*48

	fileid = H5Fcreate(file48, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	dimsdata[0] = DIML1;
	dimsdata[1] = 1;
	dimsdata[2] = 48;
	dimsdata[3] = 48;
	dimslabel[0] = DIML1;
	dimslabel[1] = 6776;
	dataspaceid = H5Screate_simple(4, dimsdata, NULL);
	datasetid = H5Dcreate(fileid, "/data48", H5T_IEEE_F64LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/label48", H5T_IEEE_F64LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(datasetid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data48);
	H5Dwrite(labelid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);
	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	status = H5Dclose(labelid);
	status = H5Sclose(labelspaceid);
	status = H5Fclose(fileid);

	//24*24

	fileid = H5Fcreate(file24, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	dimsdata[0] = DIML1;
	dimsdata[1] = 1;
	dimsdata[2] = 24;
	dimsdata[3] = 24;
	dimslabel[0] = DIML1;
	dimslabel[1] = 6776;
	dataspaceid = H5Screate_simple(4, dimsdata, NULL);
	datasetid = H5Dcreate(fileid, "/data24", H5T_IEEE_F64LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/label24", H5T_IEEE_F64LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(datasetid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data24);
	H5Dwrite(labelid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);
	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	status = H5Dclose(labelid);
	status = H5Sclose(labelspaceid);
	status = H5Fclose(fileid);
}
int main()
{
	for (int i = 35; i <= 60; i++)
	{
		st = (i - 1) * 12126 + 1;
		en = i * 12126;
		nowcnt = 0;
		sprintf(file96, "%s%d%s", file96prefix, i, ".h5");
		sprintf(file48, "%s%d%s", file48prefix, i, ".h5");
		sprintf(file24, "%s%d%s", file24prefix, i, ".h5");
		localization();
		genHDF5();
	}
	
	return 0;
}
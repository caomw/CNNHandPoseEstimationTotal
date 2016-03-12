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
#define datasetdir "D:\\CNN\\oridataours\\"
#define file96 "D:\\CNN\\trialtest96.h5"
#define file48 "D:\\CNN\\trialtest48.h5"
#define file24 "D:\\CNN\\trialtest24.h5"
#define st 1
#define en 252
#define DIML1 6776
#define DIML2 en-st+1
using namespace std;
using namespace cv;
const int sizemode[3] = { 96, 48, 24 };
double a[111][111],data96[96][96][1][en-st+1],data48[48][48][1][en-st+1],data24[24][24][1][en-st+1],label[DIML1][DIML2];
void localization()
{
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char filename[111];
		sprintf(filename, "%s%d%s", datasetdir, i, ".png");

		IplImage* src = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);   //载入灰度图像，深度为8U

		IplImage* dst = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_64F, 1);

		cvConvertScale(src, dst, 1.0 / 255, 0);
		Mat savez = Mat::zeros(96, 96, CV_64FC1);
		for (int j = 0; j < 96; j++)
		{
			for (int k = 0; k < 96; k++)
			{
				savez.at<double>(j, k) = ((double *)(dst->imageData + dst->widthStep*k))[j];
			}
		}
		for (int mode = 0; mode < 3; mode++)
		{
			int size = sizemode[mode];
			Mat z = Mat::zeros(size, size, CV_64FC1);
			resize(savez, z, Size(size, size));
			double aveall = 0;
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					aveall += z.at<double>(row, col);
				}
			}
			aveall /= (size*size);
			double stdall = 0.0;
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					stdall += pow(z.at<double>(row, col) - aveall, 2);
				}
			}
			stdall /= (size*size);
			stdall = sqrt(stdall);
			Mat z2 = z;
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
							ave += z2.at<double>(k, l);
							cnt++;
						}
					}
					ave = ave / cnt;
					double stdnow = 0;
					for (int k = max(0, row - (4 - 1 - mode)); k <= min(size - 1, row + (4 - 1 - mode)); k++)
					{
						for (int l = max(0, col - (4 - 1 - mode)); l <= min(size - 1, col + (4 - 1 - mode)); l++)
						{
							stdnow += pow(z2.at<double>(k, l) - ave, 2);
						}
					}
					stdnow = stdnow / cnt;
					stdnow = sqrt(stdnow);
					a[row][col] = (z.at<double>(row, col) - ave) / (stdnow + stdall);
					pmin = min(pmin, a[row][col]);
					pmax = max(pmax, a[row][col]);
				}
			}
			IplImage* img = cvCreateImage(cvSize(size, size), IPL_DEPTH_8U, 1);
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					((uchar *)(img->imageData + img->widthStep*col))[row] = int((a[row][col] - pmin) / (pmax - pmin)*255.0);
					switch (mode)
					{
						case 0:data96[row][col][0][i - st] = (a[row][col] - pmin) / (pmax - pmin); break;
						case 1:data48[row][col][0][i - st] = (a[row][col] - pmin) / (pmax - pmin); break;
						case 2:data24[row][col][0][i - st] = (a[row][col] - pmin) / (pmax - pmin); break;					
					}					
				}
			}
			char savefile[111];
			sprintf(savefile, "%s%d%s%d%s", "D:\\CNN\\test", sizemode[mode], "\\", i, ".png");
			cvSaveImage(savefile, img);
		}
	}
}
void genHDF5()
{
	//96*96
	hid_t fileid, datasetid, dataspaceid,labelid,labelspaceid;
	herr_t status;
	fileid = H5Fcreate(file96, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dimsdata[4],dimslabel[2];
	dimsdata[0] = 96;
	dimsdata[1] = 96;
	dimsdata[2] = 1;
	dimsdata[3] = en-st+1;
	dimslabel[0] = 6776;
	dimslabel[1] = en - st + 1;
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
	
	dimsdata[0] = 48;
	dimsdata[1] = 48;
	dimsdata[2] = 1;
	dimsdata[3] = en - st + 1;
	dimslabel[0] = 6776;
	dimslabel[1] = en - st + 1;
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
	
	dimsdata[0] = 24;
	dimsdata[1] = 24;
	dimsdata[2] = 1;
	dimsdata[3] = en - st + 1;
	dimslabel[0] = 6776;
	dimslabel[1] = en - st + 1;
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
	localization();
	genHDF5();
	return 0;
}
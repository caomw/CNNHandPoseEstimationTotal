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

#define datasetdir "J:\\cnnhandtotal\\cnntraindata\\res96\\"
#define jointprefix "J:\\cnnhandtotal\\cnntraindata\\sizeall.txt"
#define trainprefix "J:\\cnnhandtotal\\cnntraindata\\local"
#define N 72756

const int sizemode[3] = { 96, 48, 24 };
#define file96prefix "G:\\onlydepthshuffle\\trial961_"
//#define file48prefix "G:\\spatialcnnhdf5\\trial48"
//#define file24prefix "G:\\spatialcnnhdf5\\trial24"
char file96[111]/*, file48[111], file24[111]*/;
#define DIML1 12126
#define DIML2 6776
using namespace std;
using namespace cv;
int nowcnt = 0;
float ox[N + 4][15], oy[N + 4][15];
float rotx[14], roty[14];
float a[111][111], b[111][111], data96[DIML1][1][96][96]/*data96ori[DIML1][1][96][96], data48[DIML1][1][48][48], data24[DIML1][1][24][24], */,label[DIML1][DIML2];
float jntx[77777][16], jnty[77777][16];
Mat srcimg;
IplImage *img[3];
int all = 0;
const int ll[6] = { 1, 363781, 145513, 509293, 291025, 654805 };
int st, en;

int hsh[72758], sta[72758];
void init()
{
	FILE *fjoint = fopen(jointprefix, "r");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < 14; j++) fscanf(fjoint, "%f %f", &ox[i][j], &oy[i][j]);		
	}
	fclose(fjoint);
	for (int i = 1; i <= 72756; i++)
	{
		int t = (rand()*rand() + rand()) % 72756 + 1;
		while (hsh[t] == 1) t = (rand()*rand() + rand()) % 72756 + 1;
		hsh[t] = 1;
		sta[i] = t;
		//cout << i<<" "<<t << "\n ";
	}
}
void localization()
{
	for (int i = 0; i < 3; i++)
		img[i] = cvCreateImage(cvSize(sizemode[i], sizemode[i]), IPL_DEPTH_8U, 1);
	Mat z[3];
	for (int mode = 0; mode < 3; mode++)
	{
		z[mode] = Mat::zeros(sizemode[mode], sizemode[mode], CV_8UC1);
		
	}
	IplImage* src = NULL;
	
	for (int i = st; i <= en; i++)
	{
		printf("%d\n", i);
		char filename[111];
		sprintf(filename, "%s%d%s", datasetdir, sta[i],  ".png");
		//srcimg = imread(filename, 0);				
		Mat savez = imread(filename, 0);
		nowcnt++;

		


		
		int size = 96;
		resize(savez, z[0], Size(size, size));
		//imshow("", z[0]);
		//waitKey(0);

		for (int k = 0; k < 14; k++)
		{
			for (int u = 1; u <= 22; u++) //col
			{
				for (int v = 1; v <= 22; v++) //row
				{
					float t = exp(-1.0 / (2 * 1 * 1)*(pow(u - ox[all][k] - 0.5, 2) + pow(v - oy[all][k] - 0.5, 2)));
				

					if (t < 1e-6) t = 0.0;
					label[nowcnt - 1][k * 484 + (v - 1) * 22 + u - 1] = t;
				}
			}
		}
		all++;

	

		//End
		for (int row = 0; row < size; row++)
		{
			for (int col = 0; col < size; col++)
			{
				((uchar *)(img[0]->imageData + img[0]->widthStep*row))[col] =255- z[0].at<uchar>(row,col);
				data96[nowcnt - 1][0][row][col] =1.0- z[0].at<uchar>(row, col)/255.0;
			}
		}

		//waitKey(0);
		char savefile[111];
		sprintf(savefile, "%s%d%s%d%s", trainprefix, sizemode[0], "\\", nowcnt, ".png");
		cvSaveImage(savefile, img[0]);		
	}

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
	datasetid = H5Dcreate(fileid, "/data96", H5T_IEEE_F32LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/label96", H5T_IEEE_F32LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(datasetid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data96);
	H5Dwrite(labelid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);
	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	status = H5Dclose(labelid);
	status = H5Sclose(labelspaceid);
	status = H5Fclose(fileid);

}
int main()
{
	
	init();
	for (int id = 0; id < 1; id++)
	{
		for (int i = 1; i <= 72756 / DIML1; i++)
		{
			st = ll[id] - 1 + (i - 1)*DIML1 + 1;
			en = ll[id] - 1 + i*DIML1;
			nowcnt = 0;
			sprintf(file96, "%s%d%s", file96prefix, id * 72756 / DIML1 + i, ".h5");
			localization();
			genHDF5();
		}
	}
	return 0;
}//start from 6 4
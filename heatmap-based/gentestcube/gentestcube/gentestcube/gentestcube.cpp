// genexpand224.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <hdf5.h>
#include<opencv2/opencv.hpp> 
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <highgui.h>
#include<opencv2/opencv.hpp>
#define datasetdir "J:\\cnnhandtotal\\cubedata\\imgtestdep\\"
#define fileprefix "I:\\libmodel\\xyzenhance\\"
#define DIML1 8212
#define DIML2 93
#define SIZE 224
using namespace std;
using namespace cv;
float joint[72757][93];
float data[DIML1][1][SIZE][SIZE];
float label[DIML1][DIML2];
int seq[72758];
int seqexpand[145512];
int st, en, nowcnt;
char file[111];

void readgtjoint()
{
	FILE *fjoint = fopen("D:\\CNN\\jointxyztest.txt", "r");
	for (int i = 0; i < 8212; i++)
	{
	    for (int j = 0; j < 93; j++)
		{
			fscanf(fjoint, "%f", &joint[i][j]);
		}
	}
	
	fclose(fjoint);
}

void localization()
{

	Mat z;
	z = Mat::zeros(SIZE, SIZE, CV_8UC1);
	IplImage* src = NULL;
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char filename[111];
		sprintf(filename, "%s%d%s", datasetdir,i+1, ".png");
		Mat savez = imread(filename, 0);
		nowcnt++;
		resize(savez, z, Size(SIZE, SIZE));
		Mat rgb = Mat::zeros(224, 224, CV_8UC3);
		for (int row = 0; row < SIZE; row++)
		{
			for (int col = 0; col < SIZE; col++)
			{
				data[nowcnt - 1][0][row][col] = z.at<uchar>(row, col) / 255.0;
				rgb.at<Vec3b>(row, col)[0] = z.at<uchar>(row, col);
				rgb.at<Vec3b>(row, col)[1] = z.at<uchar>(row, col);
				rgb.at<Vec3b>(row, col)[2] = z.at<uchar>(row, col);
			}
		}
		for (int j = 0; j < 31; j++)
		{
			int gtu = (int)((label[nowcnt - 1][j * 3] + 1.0) * 224.0 / 2);
			int gtv = (int)((-label[nowcnt - 1][j * 3 + 1] + 1.0)*224.0 / 2.0);
			//cout << gtu << " " << gtv << "\n";
			circle(rgb, Point(gtu, gtv), 3, Scalar(0, 0, 255), -2);
		}
		imshow("", rgb);
		waitKey(1);
	}
}
void genHDF5()
{
	hid_t fileid, datasetid, dataspaceid, labelid, labelspaceid, datasetoriid, dataspaceoriid, datasetwhiteid, dataspacewhiteid;
	herr_t status;
	fileid = H5Fcreate(file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dimsdata[4], dimsdataori[2];
	hsize_t dimslabel[2];
	dimsdata[0] = DIML1;
	dimsdata[1] = 1;
	dimsdata[2] = SIZE;
	dimsdata[3] = SIZE;

	dimslabel[0] = DIML1;
	dimslabel[1] = DIML2;

	dataspaceid = H5Screate_simple(4, dimsdata, NULL);
	datasetid = H5Dcreate(fileid, "/data", H5T_IEEE_F32LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/label", H5T_IEEE_F32LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	H5Dwrite(datasetid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

	H5Dwrite(labelid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);

	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);


	status = H5Dclose(labelid);


	status = H5Sclose(labelspaceid);

	status = H5Fclose(fileid);

}
void genexpand()
{
	for (int i = 0; i < 8212 / DIML1; i++)
	{
		for (int j = 0; j < DIML1; j++)
		{
			for (int k = 0; k < 93; k++)
			{
				label[j][k] = joint[j][k];
			}

		}
		st = i*DIML1;
		en = (i + 1)*DIML1 - 1;
		nowcnt = 0;
		sprintf(file, "%s%d%s", fileprefix, i + 1, ".h5");
		localization();
		genHDF5();
	}
}
int main()
{


	readgtjoint();

	genexpand();
	return 0;
}
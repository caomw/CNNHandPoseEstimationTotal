// genliball.cpp : 定义控制台应用程序的入口点。
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
#include<hdf5.h>

#define nyudir "G:\\nyu_hand_dataset_v2\\dataset\\test\\"
#define depdir "J:\\cnnhandtotal\\cnntraindata\\nyutest\\"
#define saveprefix "G:\\"
#define DIML1 8251
#define DIML2 93
#define SIZE 224
using namespace std;
using namespace cv;
int minh, maxh, minw, maxw;
float ave0;
float portion[111111];
float data[DIML1][1][SIZE][SIZE];
float label[DIML1][DIML2];
float all[72756][DIML2];
char file[111];
int seq[111111];
IplImage *img;
int st, en, nowcnt;
void outputportion()
{
	ave0 = 754;
	FILE *fout = fopen("D:\\CNN\\genlibtest\\portion.txt", "w");
	char filenameimg[111];
	char filenameskin[111];
	for (int i = 1; i < 8252; i++)
	{
		cout << i << "\n";
		sprintf(filenameimg, "%sdepth_%d_%07d.png", nyudir, 1, i + 1);
		Mat img = imread(filenameimg);

		sprintf(filenameskin, "%ssynthdepth_%d_%07d.png", nyudir, 1, i + 1);
		Mat skin = imread(filenameskin);

		minh = 1111;
		minw = 1111;
		maxh = 0;
		maxw = 0;
		for (int h = 1; h < 480; h++)
		{
			for (int w = 1; w < 640; w++)
			{
				if (!(skin.at<Vec3b>(h, w)[0] == 0 && skin.at<Vec3b>(h, w)[1] == 0))
				{
					//cout << h << " " << w << "\n";
					minh = min(minh, h);
					maxh = max(maxh, h);
					minw = min(minw, w);
					maxw = max(maxw, w);
				}
			}
		}
		//cout << minh << " " << maxh << " " << minw << " " << maxw << "\n";
		minh = max(0, minh - 10); maxh = min(480, maxh + 10);
		minw = max(0, minw - 10); maxw = min(640, maxw + 10);
		if (maxw - minw > maxh - minh)
		{
			int delrow = maxh - minh, delcol = maxw - minw;
			int t = min(minh, (delcol - delrow) / 2);
			if (maxh + (delcol - delrow) - t > 480)
				t = maxh - 480 + (delcol - delrow);
			minh = minh - t;
			maxh = maxh + (delcol - delrow) - t;
		}
		else
		{
			int delcol = maxw - minw, delrow = maxh - minh;
			int t = min(minw, (delrow - delcol) / 2);
			if (maxw + (delrow - delcol) - t > 640)
				t = maxw - 640 + (delrow - delcol);
			minw = minw - t;
			maxw = maxw + (delrow - delcol) - t;
		}

		int sum = 0, cnt = 0;
		for (int h = minh; h <= maxh; h++)
		{
			for (int w = minw; w <= maxw; w++)
			{

				if (skin.at<Vec3b>(h, w)[0] == 0) continue;
				int depth = img.at<Vec3b>(h, w)[1] * 255 + img.at<Vec3b>(h, w)[0];
				sum += depth;
				cnt++;
			}
		}
		//imshow("", img);
		//waitKey(0);
		float ave = sum / cnt;
		
		fprintf(fout, "%.4f\n", (float)(ave0 / max(ave0, ave)));
	}
	fclose(fout);
}
void init()
{
	FILE *fin = fopen("D:\\CNN\\genlibtest\\portion.txt", "r");
	for (int i = 0; i < 8251; i++) fscanf(fin, "%f", &portion[i]);
	fclose(fin);
	fin = fopen("J:\\cnnhandtotal\\libhandmodelexp\\xyzpred224.txt", "r");
	
	
	for (int i = 0; i < 8251; i++)
	{
		for (int j = 0; j < 93; j++)
		{
			fscanf(fin, "%f", &all[i][j]);
		}
		//change percentage
		for (int j = 0; j < 31; j++)
		{
			//all[i][3 * j] = (all[i][3 * j] - SIZE / 100.0 / 2)*portion[i] + SIZE / 100.0 / 2;
			//all[i][3 * j + 1] = (all[i][3 * j + 1] - SIZE / 100.0 / 2)*portion[i] + SIZE / 100.0 / 2;
		}
	}
	fclose(fin);
}
void localization(int opt)
{
	img = cvCreateImage(cvSize(SIZE, SIZE), IPL_DEPTH_8U, 1);
	Mat z;
	z = Mat::zeros(SIZE, SIZE, CV_8UC1);
	IplImage* src = NULL;
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		if (i == 0)
		{
			int t = 1;
		}
		char filename[111];		
		sprintf(filename, "%s%d%s", depdir, i+2, ".png");
		Mat savez = imread(filename, 0);
		Mat z = Mat::zeros(Size(224, 224), CV_8UC1);
		resize(savez, z, Size(224, 224));
		nowcnt++;
		imshow("", z);
		waitKey(1);
		for (int row = 0; row < SIZE; row++)
		{
			for (int col = 0; col < SIZE; col++)
			{
				//cout << " " << row << " " << col << "\n";
				data[nowcnt - 1][0][row][col] =z.at<uchar>(row, col) / 255.0;
			}
		}
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
void genorigindep()
{
	for (int j = 0; j < DIML1; j++)
	{
		for (int k = 0; k < 93; k++)
		{
			label[j][k] = all[j][k];
		}
	}
	st = 0;
	en = 8250; //2-8252		
	nowcnt = 0;
	sprintf(file, "%s%s", saveprefix, "nyutest.h5");
	localization(1);
	genHDF5();
	
}
int main()
{
	//outputportion();
	init();
	genorigindep();
	
	return 0;
}
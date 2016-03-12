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
#define DIML1 6063
#define DIML2 93
#define SIZE 224
#define nyudir "G:\\nyu_hand_dataset_v2\\dataset\\train\\"
#define depdir "J:\\cnnhandtotal\\cnntraindata\\size224\\dep\\"
#define depprefix "H:\\libmodel\\xyzdirectdep\\"
using namespace std;
using namespace cv;
int minh, maxh, minw, maxw;
float ave0;
float portion[111111];
float data[DIML1][1][SIZE][SIZE];
float label[DIML1][DIML2];
float all[72756][DIML2];
float gtz[33];
char file[111];
int seq[111111];
IplImage *img;
int st, en, nowcnt;
void init()
{
	FILE *fin = fopen("D:\\CNN\\genliball\\portion.txt", "r");
	for (int i = 0; i < 72756; i++) fscanf(fin, "%f", &portion[i]);
	fclose(fin);
	
	FILE *fseq = fopen("D:\\CNN\\genlibmodeltrainHDF5\\seq.txt", "r");
	for (int i = 0; i < 72756; i++) { fscanf(fseq, "%d", &seq[i]); } //id:1-72756
	fclose(fseq);

	FILE *fingt = fopen("J:\\cnnhandtotal\\libhandmodelexp\\xyzgt.txt", "r");
	FILE *finuv = fopen("J:\\cnnhandtotal\\libhandmodelexp\\xyz224.txt", "r");

	char filenameimg[111];
	char filenameskin[111];
	for (int i = 0; i < 1; i++)
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
		float ave = sum / cnt;
		float std = 0.0;
		for (int j = 0; j < 31; j++)
		{
			fscanf(fingt, "%f", &gtz[j]);
			std += pow(gtz[j] - ave, 2);
		}
		
		std = sqrt(std);
		for (int j = 0; j < 31; j++)
		{
			gtz[j] = 2.0 + (gtz[j] - ave) / std;
			//cout << gtz[j] << " ";
		}
		//cout << "\n";
		
		for (int j = 0; j < 93; j++)
		{
			fscanf(finuv, "%f", &all[i][j]);
		}
		FILE *foutinit = fopen("D:\\CNN\\handmodel\\libmodel\\initialnew.txt", "w");
		for (int j = 0; j < 31; j++)
		{
			float z = gtz[j];
			float x = (all[i][j * 3] * 100.0 / SIZE - 0.5)*z;
			float y = (0.5 - all[i][j * 3 + 1] * 100.0 / SIZE)*z;
			all[i][j * 3] = x;
			all[i][j * 3 + 1] = y;
			all[i][j * 3 + 2] = z;
			//cout << " " << x << " " << y << " " << z << "\n";
			fprintf(foutinit, "%.4f %.4f %.4f\n", x, y, z);
		}
		fclose(foutinit);
		//imshow("", img);
		//waitKey(0);
		
	}
	
	fclose(fingt);
	fclose(finuv);
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
		char filename[111];
		sprintf(filename, "%s%d%s", depdir, seq[i], ".png");
		Mat z = imread(filename, 0);
		nowcnt++;

		for (int row = 0; row < SIZE; row++)
		{
			for (int col = 0; col < SIZE; col++)
			{
				data[nowcnt - 1][0][row][col] = z.at<uchar>(row, col) / 255.0;
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
	for (int i = 0; i < 72756 / DIML1; i++)
	{
		for (int j = 0; j < DIML1; j++)
		{
			for (int k = 0; k < 93; k++)
			{
				label[j][k] = all[seq[i*DIML1 + j] - 1][k];
			}

		}
		st = i*DIML1;
		en = (i + 1)*DIML1 - 1;
		nowcnt = 0;
		sprintf(file, "%s%d%s", depprefix, i + 1, ".h5");
		localization(1);
		genHDF5();
	}
}
int main()
{
	init();
	//genorigindep();
	return 0;
}


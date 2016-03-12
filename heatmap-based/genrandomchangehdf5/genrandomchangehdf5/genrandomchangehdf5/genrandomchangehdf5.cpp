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
#define datasetdir "G:\\spatialcnndata\\"
#define jointprefix "D:\\ori.txt"
#define trainprefix "G:\\local"
int st, en;
const int sizemode[3] = { 96, 48, 24 };
#define file96prefix "G:\\spatialcnnhdf5\\trial96"
//#define file48prefix "G:\\spatialcnnhdf5\\trial48"
//#define file24prefix "G:\\spatialcnnhdf5\\trial24"
char file96[111]/*, file48[111], file24[111]*/ ;
#define DIML1 6063
#define DIML2 6776
using namespace std;
using namespace cv;
using namespace Eigen;
int nowcnt = 0;
double ox[14], oy[14];
double rotx[14], roty[14];
double a[111][111], b[111][111],data96[DIML1][1][96][96],data96ori[DIML1][1][96][96], /*data48[DIML1][1][48][48], data24[DIML1][1][24][24], */label[DIML1][DIML2];
double jntx[77777][16],jnty[77777][16];
Mat srcimg;
IplImage *img[3];
void initjointfile()
{
	FILE *fjoint = fopen(jointprefix, "r");
	for (int i = 1; i <= 72756; i++)
	{		
		for (int j = 0; j < 14; j++)
		{
			fscanf(fjoint, "%lf %lf", &jntx[i][j], &jnty[i][j]);
		}
	}
	fclose(fjoint);
}
void localization(int op,int latter)
{
	for (int i = 0; i < 3; i++)
		img[i] = cvCreateImage(cvSize(sizemode[i], sizemode[i]), IPL_DEPTH_8U, 1);	
	Mat z[3],zori[3];
	Mat z2;
	for (int mode = 0; mode < 3; mode++)
	{
		z[mode] = Mat::zeros(sizemode[mode], sizemode[mode], CV_8UC1);
		zori[mode] = Mat::zeros(sizemode[mode], sizemode[mode], CV_8UC1);
	}
	IplImage* src = NULL;	
	st = (latter - 1)*DIML1 + 1; en = latter*DIML1;
	for (int i = st; i <= en; i++)
	{
		cout << i <<" "<< op<<" "<<latter<<"\n";
		char filename[111];
		sprintf(filename, "%s%d%s%d%s", datasetdir, i,"_",op, ".png");		
		//srcimg = imread(filename, 0);				
		Mat savez = imread(filename, 0);
		nowcnt++;
		
		char filenameori[111];
		sprintf(filenameori, "%s%d%s%d%s", datasetdir, i, "_", 1, ".png");
		Mat savezori = imread(filenameori, 0);

		for (int k = 0; k < 14; k++)
		{
			for (int u = 1; u <= 22; u++)
			{
				for (int v = 1; v <= 22; v++)
				{
					label[nowcnt - 1][k * 484 + u * 22 + v] = exp(-1.0 / (2 * 1 * 1)*(pow(u - jntx[i][k] - 0.5, 2) + pow(v - jnty[i][k] - 0.5, 2)));
				}
			}
		}


		for (int mode = 0; mode < 1; mode++)
		{
			int size = sizemode[mode];

			resize(savez, z[mode], Size(size, size));
			//imshow("", z[mode]);
			//waitKey(0);
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
			//Origin
			resize(savezori, zori[mode], Size(size, size));
			//imshow("", z[mode]);
			//waitKey(0);
			for (int row = 0; row < size; row++)
			{
				for (int col = row; col < size; col++)
				{
					int t = zori[mode].at<uchar>(row, col);
					zori[mode].at<uchar>(row, col) = zori[mode].at<uchar>(col, row);
					zori[mode].at<uchar>(col, row) = t;
				}
			}
			//imshow("", z[mode]);
			//waitKey(0);
			double aveallori = 0;
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					aveallori += zori[mode].at<uchar>(row, col);
				}
			}
			aveallori /= (size*size);
			double stdallori = 0.0;
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					stdallori += pow(zori[mode].at<uchar>(row, col) - aveallori, 2);
				}
			}
			stdallori /= (size*size);
			stdallori = sqrt(stdallori);
			z2 = zori[mode];
			double pminori = 1111.0, pmaxori = 0.0;
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					double aveori = 0;
					int cnt = 0;
					for (int k = max(0, row - (4 - 1 - mode)); k <= min(size - 1, row + (4 - 1 - mode)); k++)
					{
						for (int l = max(0, col - (4 - 1 - mode)); l <= min(size - 1, col + (4 - 1 - mode)); l++)
						{
							aveori += z2.at<uchar>(k, l);
							cnt++;
						}
					}
					aveori = aveori / cnt;
					double stdnowori = 0;
					for (int k = max(0, row - (4 - 1 - mode)); k <= min(size - 1, row + (4 - 1 - mode)); k++)
					{
						for (int l = max(0, col - (4 - 1 - mode)); l <= min(size - 1, col + (4 - 1 - mode)); l++)
						{
							stdnowori += pow(z2.at<uchar>(k, l) - aveori, 2);
						}
					}
					stdnowori = stdnowori / cnt;
					stdnowori = sqrt(stdnowori);
					b[row][col] = (zori[mode].at<uchar>(row, col) - aveori) / (stdnowori + stdallori);
					pminori = min(pminori, b[row][col]);
					pmaxori = max(pmaxori, b[row][col]);
				}
			}
			//End
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					//((uchar *)(img[mode]->imageData + img[mode]->widthStep*row))[col] = int((b[row][col] - pminori) / (pmaxori - pminori)*255.0);
					switch (mode)
					{
					case 0:data96[nowcnt - 1][0][row][col] = (a[row][col] - pmin) / (pmax - pmin); 
						   data96ori[nowcnt - 1][0][row][col] = (b[row][col] - pminori) / (pmaxori - pminori);
						break;
					//case 1:data48[nowcnt - 1][0][row][col] = (a[row][col] - pmin) / (pmax - pmin); break;
					//case 2:data24[nowcnt - 1][0][row][col] = (a[row][col] - pmin) / (pmax - pmin); break;
					}
				}
			}

			waitKey(0);
			char savefile[111];
			sprintf(savefile, "%s%d%s%d%s", trainprefix, sizemode[mode], "\\", nowcnt, ".png");
			//cvSaveImage(savefile, img[mode]);
		}
	}
	
}
void genHDF5()
{
	//96*96
	hid_t fileid, datasetid, dataspaceid, datasetoriid,dataspaceoriid,   labelid, labelspaceid;
	herr_t status;
	fileid = H5Fcreate(file96, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dimsdata[4],dimsoridata[4], dimslabel[2];
	dimsdata[0] = DIML1;
	dimsdata[1] = 1;
	dimsdata[2] = 96;
	dimsdata[3] = 96;

	dimsoridata[0] = DIML1;
	dimsoridata[1] = 1;
	dimsoridata[2] = 96;
	dimsoridata[3] = 96;

	dimslabel[0] = DIML1;
	dimslabel[1] = 6776;
	dataspaceid = H5Screate_simple(4, dimsdata, NULL);
	datasetid = H5Dcreate(fileid, "/data96", H5T_IEEE_F64LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	dataspaceoriid = H5Screate_simple(4, dimsoridata, NULL);
	datasetoriid = H5Dcreate(fileid, "/dataori96", H5T_IEEE_F64LE, dataspaceoriid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/label96", H5T_IEEE_F64LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(datasetid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data96);
	H5Dwrite(datasetoriid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data96ori);
	H5Dwrite(labelid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);
	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	status = H5Dclose(datasetoriid);
	status = H5Sclose(dataspaceoriid);
	status = H5Dclose(labelid);
	status = H5Sclose(labelspaceid);
	status = H5Fclose(fileid);
	
	//48*48

	/*fileid = H5Fcreate(file48, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

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
	status = H5Fclose(fileid);*/

	//24*24

	/*fileid = H5Fcreate(file24, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

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
	status = H5Fclose(fileid);*/
}
int main()
{
	initjointfile();
	for (int i = 7; i <= 10; i++)
	{		
		
		for (int j = 1; j <= 12; j++)
		{
			nowcnt = 0;
			sprintf(file96, "%s%d%s%d%s", file96prefix, i,"_",j, ".h5");
			//sprintf(file48, "%s%d%s%d%s", file48prefix, i,"_",j, ".h5");
			//sprintf(file24, "%s%d%s%d%s", file24prefix, i,"_",j, ".h5");
			localization(i,j);
			genHDF5();
		}
		
	}

	return 0;
}//start from 6 4
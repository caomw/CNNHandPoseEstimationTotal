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
#define datasetdir "D:\\CNN\\kinectdata\\0824\\res96\\"
//#define datasetdir "F:\\cnnhandtotal\\rdfcnn\\picgray\\"
#define jointprefix "D:\\CNN\\kinectdata\\0824\\"
//#define jointprefix "F:\\cnnhandtotal\\rdfcnn\\joint\\joint\\"
#define trainprefix "D:\\CNN\\kinectdata\\0824\\train"
//#define trainprefix "F:\\cnnhandtotal\\rdfcnn\\train"
#define st 863
#define en 1672
const int angle[13] = { -5, 0, 5, -8, 8, -10, 10, -12, 12, -15, 15, -18, 18};
const int sizemode[3] = { 96, 48, 24 };
#define file96 "D:\\CNN\\handjoint\\newtrain\\trial9624_2.h5"
#define file48 "D:\\CNN\\handjoint\\newtrain\\trial4824_2.h5"
#define file24 "D:\\CNN\\handjoint\\newtrain\\trial2424_2.h5"
#define DIML1 (en-st+1)*13
#define DIML2 6776
using namespace std;
using namespace cv;
using namespace Eigen;
int jx[14], jy[14],nowcnt=0;
double ox[14], oy[14];
double rotx[14],roty[14];
double a[111][111], data96[DIML1][1][96][96], data48[DIML1][1][48][48], data24[DIML1][1][24][24], label[DIML1][DIML2];
Mat srcimg, rotimg;
MatrixXd A(2,2);
float m[6];
IplImage *img[3];
void rotateImage(Mat img, Mat img_rotate, int degree)
{
	//Ðý×ªÖÐÐÄÎªÍ¼ÏñÖÐÐÄ  
	Point2f center;
	center.x = float(96 / 2.0 + 0.5);
	center.y = float(96 / 2.0 + 0.5);
	//¼ÆËã¶þÎ¬Ðý×ªµÄ·ÂÉä±ä»»¾ØÕó  
	Mat M = Mat(2, 3, CV_32F, m);
	M = getRotationMatrix2D(center, degree, 1);	
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			m[i * 3 + j] = M.at<double>(i, j);
			if (j<2) A(i, j) = m[i * 3 + j];
		}
	}
	

	//±ä»»Í¼Ïñ£¬²¢ÓÃºÚÉ«Ìî³äÆäÓàÖµ  
	warpAffine(img, img_rotate, M, Size(96, 96), 1, BORDER_CONSTANT, Scalar(255,255,255));
	//imshow("", img_rotate);
	//waitKey(0);
}
void localization()
{
	for (int i = 0; i < 3; i++)
		img[i] = cvCreateImage(cvSize(sizemode[i], sizemode[i]), IPL_DEPTH_8U, 1);
	IplImage* dst = cvCreateImage(cvSize(96, 96), IPL_DEPTH_64F, 1);
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char filename[111];
		sprintf(filename, "%s%d%s", datasetdir, i, ".png");
		//char picname[111];
		//sprintf(picname, "%s%d%s", "F:\\cnnhandtotal\\rdfcnn\\showjointonpicture\\", i, ".png");
		//FILE *ftest = fopen(picname, "r"); //not exist
		//if (ftest == NULL) continue;
		//fclose(ftest);
		for (int j = 0; j < 14; j++)
		{
			char jointname[111];
			sprintf(jointname, "%s%d%s%d%s", jointprefix, j, "\\", i, ".txt");
			//sprintf(jointname, "%s%d%s%d%s", jointprefix, i, "_", j, ".txt");
			FILE *fin = fopen(jointname, "r");
			//fscanf(fin, "%d %d", &jy[j], &jx[j]); //column row
			fscanf(fin, "%lf %lf", &oy[j], &ox[j]);
			ox[j] = ox[j] / 500.0*96.0; oy[j] = oy[j] / 500.0*96.0;
			fclose(fin);
		}
		srcimg = imread(filename, 0);
		rotimg = srcimg;
		for (int j = 0; j < 13; j++)
		{
			nowcnt++;
			rotimg = srcimg;
			rotateImage(srcimg, rotimg, angle[j]);
			imwrite("D:\\tmp.png", rotimg);
			for (int k = 0; k < 14; k++)
			{
				VectorXd h;
				VectorXd x(2);
				x[0] = ox[k] - m[2];
				x[1] = oy[k] - m[5]; // [ M11 M12 M13    [ x       [ x0
				//   M21 M22 M23 ]	   y    =    y0 ]
				//                    1 ] 								 
				h = (A.inverse())*x;
				rotx[k] = h[0] / 96.0*22.0; roty[k] = h[1] / 96.0*22.0; //x y    row col
				//cout << jx[k] /500.0*22.0<< " " << jy[k]/500.0*22.0 << " " << rotx[k] << " " << roty[k] << "\n";
				//cout << ox[k] << " " << oy[k] << " " << rotx[k] / 22.0*96.0 << " " << roty[k] / 22.0*96.0 << "\n";
				for (int u = 1; u <= 22; u++)
				{
					for (int v = 1; v <= 22; v++)
					{
						label[nowcnt - 1][k*484+u*22+v]= exp(-1.0 / (2 * 1 * 1)*(pow(u - roty[k] - 0.5, 2) + pow(v - rotx[k] - 0.5,2)));
					}
				}
			}
			IplImage* src = cvLoadImage("D:\\tmp.png", CV_LOAD_IMAGE_GRAYSCALE);   //ÔØÈë»Ò¶ÈÍ¼Ïñ£¬Éî¶ÈÎª8U
			cvConvertScale(src, dst, 1.0);
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

				for (int row = 0; row < size; row++)
				{
					for (int col = 0; col < size; col++)
					{
						((uchar *)(img[mode]->imageData + img[mode]->widthStep*col))[row] = int((a[row][col] - pmin) / (pmax - pmin)*255.0);
						switch (mode)
						{
						case 0:data96[nowcnt-1][0][row][col] = (a[row][col] - pmin) / (pmax - pmin); break;
						case 1:data48[nowcnt-1][0][row][col] = (a[row][col] - pmin) / (pmax - pmin); break;
						case 2:data24[nowcnt-1][0][row][col] = (a[row][col] - pmin) / (pmax - pmin); break;
						}
					}
				}
				char savefile[111];
				sprintf(savefile, "%s%d%s%d%s", trainprefix, sizemode[mode], "\\", nowcnt, ".png");
				cvSaveImage(savefile, img[mode]);
			}
		}
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
	localization();
	genHDF5();
	return 0;
}
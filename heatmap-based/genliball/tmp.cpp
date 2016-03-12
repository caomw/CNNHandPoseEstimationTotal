#define synthdir "J:\\cnnhandtotal\\cnntraindata\\size224\\synthdep\\"
#define depdir "J:\\cnnhandtotal\\cnntraindata\\size224\\dep\\"
void init()
{
	FILE *fin = fopen("J:\\cnnhandtotal\\libhandmodelexp\\xyz224.txt", "r");
	FILE *fseq = fopen("D:\\CNN\\genlibmodeltrainHDF5\\seq.txt", "r");
	for (int i = 0; i < 72756; i++) { fscanf(fseq, "%d", &seq[i]);  } //id:1-72756
	fclose(fseq);
	for (int i = 0; i < 72756; i++)
	{
		for (int j = 0; j < 93; j++)
		{
			fscanf(fin, "%f", &all[i][j]);			
		}
			
	}
	fclose(fin);
}
void localization()
{	
	img = cvCreateImage(cvSize(SIZE, SIZE), IPL_DEPTH_8U, 1);
	Mat z;	
	z = Mat::zeros(SIZE, SIZE, CV_8UC1);	
	IplImage* src = NULL;
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char filename[111];
		sprintf(filename, "%s%d%s", datasetdir, seq[i], ".png");		
		Mat savez = imread(filename, 0);
		nowcnt++;		
		resize(savez, z, Size(SIZE,SIZE));
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





#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<iostream>
#include<ctime>
#define NUM1 4
#define NUM2 2
using namespace std;
int a[11111111],ind[1111111],hsh[1111111],cnt;
int main()
{
	srand(time(0));
	FILE *fout=fopen("D:\\CNN\\genlibmodeltrainHDF5\\seqexpand.txt","w");
	FILE *fin=fopen("D:\\CNN\\genliball\\rankerr.txt","r");
	for (int i=0;i<72756;i++)
	{
		int id; 
		double err;
		fscanf(fin,"%d %lf",&id,&err);
		if (err>=10.0 && err<=20.0)
		{
			for (int j=0;j<NUM1;j++)
			{
				a[cnt++]=id;
			}
		}
		else if (err>20.0)
		{
			for (int j=0;j<NUM2;j++)
			{
				a[cnt++]=id;
			}
		}
	}
	for (int i=0;i<cnt;i++)
	{
		int t=(rand()*rand()+rand())%cnt;
		while (hsh[t]==1) t=(rand()*rand()+rand())%cnt;
		hsh[t]=1;
		ind[i]=t;	
	}	
	for (int i=0;i<cnt;i++)
	{
		fprintf(fout,"%d\n",a[ind[i]]);
	}
	fclose(fin);
	fclose(fout);
	return 0;
}

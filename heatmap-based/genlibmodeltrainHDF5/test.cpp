#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<iostream>
#include<ctime>
using namespace std;
int hsh[100000];
int main()
{
	FILE *fout=fopen("D:\\CNN\\genlibmodeltrainHDF5\\seq.txt","w");
	for (int i=0;i<72756;i++)
	{
		int t=(rand()*rand()+rand())%72756+1;
		while (hsh[t]) t=(rand()*rand()+rand())%72756+1;
		hsh[t]=1;
		fprintf(fout,"%d\n",t);
	}
	
	fclose(fout);
	return 0;	
}

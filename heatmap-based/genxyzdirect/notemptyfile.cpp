#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<cmath>
#include<iostream>
#define prefix "I:\\hardposes\\"
using namespace std;
int main()
{
	char filename[111];
	FILE *fout=fopen("D:\\CNN\\genlibmodeltrainHDF5\\seqhard.txt","w");
	for (int i=1;i<=72756;i++)
	{
		sprintf(filename,"%s%d%s",prefix,i,".png");
		FILE *ftmp=fopen(filename,"r");
		if (ftmp!=NULL)
		{
			fprintf(fout,"%d\n",i);
		}
		fclose(ftmp);
	}
	
	return 0;
}

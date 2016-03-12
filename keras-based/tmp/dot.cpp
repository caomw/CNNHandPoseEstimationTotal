#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<iostream>
#define BATCHSIZE 32
using namespace std;
int main()
{
	FILE *fout=fopen("D:\\CNN\\handmodel\\dot.txt","w");
	for (int i=0;i<BATCHSIZE;i++)
	{
		fprintf(fout,"        %s%d%s%d%s%d%s\n","c=T.set_subtensor(c[",i,",:],T.dot(a[",i,",:],b[",i,",:]))");
	}
	fclose(fout);
}

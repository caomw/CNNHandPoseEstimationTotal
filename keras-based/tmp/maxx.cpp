#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<iostream>
using namespace std;
int main()
{
	FILE *fout=fopen("D:\\CNN\\handmodel\\maxx.txt","w");
	for (int k=0;k<48;k++)
	{
		fprintf(fout,"        %s%d%s%d%s\n","results=T.set_subtensor(results[:,",k,",:,:],results",k+1,")");
		
	}
	fclose(fout);
}

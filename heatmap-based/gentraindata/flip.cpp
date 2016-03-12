#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<iostream>
using namespace std;
int main()
{
	FILE *fin=fopen("F:\\cnnhandtotal\\cnntraindata\\alljoint.txt","r");
	FILE *fout=fopen("F:\\cnnhandtotal\\cnntraindata\\alljoint2.txt","w");
	for (int i=0;i<363780;i++)
	{
		cout<<i<<"\n";
		for (int j=0;j<14;j++)
		{
			double x,y;
			fscanf(fin,"%lf %lf",&x,&y);
			fprintf(fout,"%.4f %.4f ",22.0-x,y);
		}
		fprintf(fout,"\n");
	}
	fclose(fin);
	fclose(fout);
	return 0;
}

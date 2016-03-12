#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<iostream>
#include<algorithm>
using namespace std;
float ans=0;
int main()
{
	FILE *fin=fopen("L:\\error.txt","r");
	for (int i=0;i<72756;i++)
	{
		float t;
		fscanf(fin,"%f",&t);
		ans+=t;
	}
	ans/=72756.0;
	printf("%.4f\n",ans);
	fclose(fin);
	return 0;
}

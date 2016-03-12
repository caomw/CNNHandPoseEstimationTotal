#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<algorithm>
#include<iostream>
using namespace std;
int seq[111111];
struct node
{
	int id;
	float key;
	node(){}
	node(int id,float key):id(id),key(key){}
}err[111111];
bool cmp(node u,node v)
{
	return u.key>v.key;
}
int cnt[1111];
int main()
{
	FILE *ferr=fopen("error.txt","r");
	FILE *fseq=fopen("D:\\CNN\\genlibmodeltrainHDF5\\seq.txt","r"); //id starting from 1
	for (int i=0;i<72756;i++)
	{
		fscanf(fseq,"%d",&seq[i]);
		seq[i]--; // convert to 0
	}
	for (int i=0;i<72756;i++)
	{
		err[seq[i]].id=seq[i];
		fscanf(ferr,"%f",&err[seq[i]].key);
	}
	sort(err,err+72756,cmp);
	FILE *fout=fopen("rankerr.txt","w");
	for (int i=0;i<72756;i++)
	{
		fprintf(fout,"%d %.2f\n",err[i].id,err[i].key);	 
		cnt[(int)err[i].key]++;
	}
	for (int i=0;i<60;i++)
	{
		cout<<i<<" "<<cnt[i]<<"\n";
	}
	fclose(fout);
	fclose(fseq);
	fclose(ferr);
	return 0;
}

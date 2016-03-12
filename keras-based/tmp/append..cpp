#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<iostream>
using namespace std;
int main()
{
	FILE *fout=fopen("D:\\CNN\\handmodel\\append.txt","w");
	for (int k=0;k<48;k++)
	{
		fprintf(fout,"        %s%d%s%s%d%s%d%s\n","ans=T.set_subtensor(ans[:,",k,"*2","],   ( ret[:,3*",k,"]    / -ret[:,3*",k,"+2]+0.5)*self.width)");
		fprintf(fout,"        %s%d%s%s%d%s%d%s\n","ans=T.set_subtensor(ans[:,",k,"*2+1","], ( -ret[:,3*",k,"+1] / -ret[:,3*",k,"+2]+0.5)*self.height)");
	}
	fclose(fout);
}

#include <stdio.h>

void ref12_3_1(/*float inp[12],*/ float *output);


int main(){


	float inp[12] = {1,1,1,1,1,1,1,1,1,1,1,1};
	float output[1000];

	ref12_3_1(/*inp,*/ output);

	printf("\noutput: %f\n", output[1]);

	/*printf("\n");

	for(int i = 0; i < 12 * 10000; i++){
		printf("%f,", (float)i / 10000);
	}*/

}




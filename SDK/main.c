
/*
 * ap_ctrl_hs ap_ready dataflow
 */

#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xil_cache.h"

#include "xref_top.h"
#include <xtime_l.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "Layer2_weights.h"
#include "Layer3_weights.h"
#include "Layer4_weights.h"
#include "Layer5_weights.h"
#include "Layer2_bias.h"
#include "Layer3_bias.h"
#include "Layer4_bias.h"
#include "Layer5_bias.h"


void port2loc(float inp[784/* * 1000*/], float inp_local[784]){
	for(int init_local = 0; init_local < 784; init_local++){
		inp_local[init_local] = inp[init_local]/*inp[dataflow_loop * 784 + init_local]*/;
	}
}

void stage0(float inp_local[784], float Layer2_weights[40][784], float Layer2_bias[40], float Layer2_neurons[40]){
	float sum;
	for(int i = 0 ; i < 40 ; i++){
		sum = 0;
		for(int j = 0 ; j < 784 ; j++)
			sum += inp_local[j] * Layer2_weights[i][j];
		Layer2_neurons[i] = (sum + Layer2_bias[i]) > 0 ? (sum + Layer2_bias[i]) : 0;
	}
}


void stage1(float Layer2_neurons[40], float Layer3_weights[40][40], float Layer3_bias[40], float Layer3_neurons[40]){
	float sum;
	for(int i = 0 ; i < 40 ; i++){
		sum = 0;
		for(int j = 0 ; j < 40 ; j++)
			sum += Layer2_neurons[j] * Layer3_weights[i][j];
		Layer3_neurons[i] = (sum + Layer3_bias[i]) > 0 ? (sum + Layer3_bias[i]) : 0;
	}
}
void stage2(float Layer3_neurons[40], float Layer4_weights[40][40], float Layer4_bias[40], float Layer4_neurons[40]){
	float sum;
	float sum_p[4] = {0,0,0,0};
	for(int i = 0 ; i < 40 ; i++){
		sum = 0;
		for(int j = 0 ; j < 40 ; j++)
			sum += Layer3_neurons[j] * Layer4_weights[i][j];
		Layer4_neurons[i] = (sum + Layer4_bias[i]) > 0 ? (sum + Layer4_bias[i]) : 0;
	}
}
void stage3(float Layer4_neurons[40], float Layer5_weights[10][40], float Layer5_bias[10], float out[10]){
	float sum;
	for(int i = 0 ; i < 10 ; i++){
		sum = 0;
		for(int j = 0 ; j < 40 ; j++)
			sum += Layer4_neurons[j] * Layer5_weights[i][j];
		out[i] = sum + Layer5_bias[i];
	}
}

void parkingNN_sw(float inp[784] , float out[10]){


	float inp_local[784];
	float Layer2_neurons[40];
	float Layer3_neurons[40];
	float Layer4_neurons[40];

	port2loc(inp, inp_local);
	stage0(inp_local, Layer2_weights, Layer2_bias, Layer2_neurons);
	stage1(Layer2_neurons, Layer3_weights, Layer3_bias, Layer3_neurons);
	stage2(Layer3_neurons, Layer4_weights, Layer4_bias, Layer4_neurons);
	stage3(Layer4_neurons, Layer5_weights, Layer5_bias, out);

}


XRef_top parkNN;

int main()
{
    init_platform();


    Xil_DCacheDisable();

    print("Entering\n\r");

    float a = 1.0;

    XTime tStart, tEnd;
    double  ElapsedTime;

    int i, j, status;
	status = XRef_top_Initialize(&parkNN,0);
	if(status != XST_SUCCESS){
	   print("HLS parkNN failed\n\r");
	   return -1;
	}


    float *Din_AXI = (float *)(0x10000000);
    float *Dout_AXI_SW = (float *)(0x13000000);
    float *DinArr = (float *)(0x2000000);
    float *Dout_SW = (float *)(0x15000000);
    float *Dout_HW = (float *)(0x16000000);

    XRef_top_Set_inp(&parkNN,Din_AXI);
    //XRef_top_Set_out_r(&parkNN,Dout_AXI);
    XRef_top_Set_out_r(&parkNN,Dout_HW);
    //XRef_top_EnableAutoRestart(&parkNN);


    print("Hello World\n\r");

    for (j=0;j<1000;j++){
		for(i=0;i<784;i++){
			DinArr[j*784+i]= ((float)rand()/(float)(RAND_MAX)) * a;
		}
    }


    for (j=0;j<1000;j++){
    	for(i=0;i<784;i++){
			Din_AXI[j*784+i]= DinArr[j*784+i];
		}
    }
    // HW Execution
	XTime_GetTime(&tStart);
	XRef_top_Start(&parkNN);
	while (!XRef_top_IsReady(&parkNN));
	XTime_GetTime(&tEnd);

    printf("HW Acc Output took %llu clock cycles.\n", 2*(tEnd - tStart));
    ElapsedTime = 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND);
    printf("HW Acc Output took %.2f s.\n",ElapsedTime);


    // SW Execution
    XTime_GetTime(&tStart);

	for (j=0;j<1000;j++){

		parkingNN_sw(&DinArr[j*784],Dout_AXI_SW);

		for(i=0;i<10;i++){
			Dout_SW[j*10+i]=Dout_AXI_SW[i];
		}
	}

	XTime_GetTime(&tEnd);

	printf("SW Acc Output took %llu clock cycles.\n", 2*(tEnd - tStart));
	ElapsedTime = 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND);
	printf("SW Acc Output took %.2f s.\n",ElapsedTime);


	int err=0;
	float err_range = 0.0001;
	for (j=0;j<1000;j++){
		for(i=0;i<10;i++){
			if(abs(Dout_SW[j*10+i] - Dout_HW[j*10+i]) > err_range){
				err++;
			}
		}
	}
	printf("error=%d\n\r", err);

	XRef_top_DisableAutoRestart(&parkNN);

    print("Exiting\n\r");

    cleanup_platform();
    return 0;
}

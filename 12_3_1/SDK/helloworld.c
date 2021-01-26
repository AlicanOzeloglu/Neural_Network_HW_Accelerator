/******************************************************************************
*
* Copyright (C) 2009 - 2014 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* Use of the Software is limited solely to applications:
* (a) running on a Xilinx device, or
* (b) that interact with a Xilinx device through a bus or interconnect.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xil_cache.h"

#include "xref12_3_1.h"
#include <xtime_l.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "inp.h"

#define dataflow 1000

void stage0(float inp[12], float weight[3][12], float out[3], float bias[3]){
	float sum;
	float sump[4];
	for(int i = 0; i < 3; i++){
		sum = 0;
		for(int j = 0; j < 12; j+=4){
			sump[0] = inp[j] * weight[i][j];
			sump[1] = inp[j+1] * weight[i][j+1];
			sump[2] = inp[j+2] * weight[i][j+2];
			sump[3] = inp[j+3] * weight[i][j+3];
			sum += sump[0] + sump[1] + sump[2] + sump[3];
		}
		out[i] = sum + bias[i];
	}

}

void stage1(float inp[3], float weight[3], float *out, float *bias){
	float sum = 0;
	for(int j = 0; j < 3; j++){
		sum += inp[j] * weight[j];
	}
	*out = sum + *bias;
}

void localMemLoad(float inp[12], float local_inp[12]){
	for(int i = 0; i < 12; i++){
		local_inp[i] = inp[i];
	}

}

void ref12_3_1(float output[dataflow]){

	float local_inp[12];

	float Layer2_weights[3][12] = {
			{1,1,1,1,1,1,1,1,1,1,1,1},
			{1,1,1,1,1,1,1,1,1,1,1,1},
			{1,1,1,1,1,1,1,1,1,1,1,1}
	};
	float Layer3_weights[3] = {1,1,1};
	float hidden[3];
	float Layer2_bias[3] = {1,1,1};
	float Layer3_bias = 1;

	for(int dataflow_loop = 0; dataflow_loop < dataflow; dataflow_loop++){
		localMemLoad(&inp[12 * dataflow_loop], local_inp);
		stage0(local_inp/*&inp[12 * dataflow_loop]*/, Layer2_weights, hidden, Layer2_bias);
		stage1(hidden, Layer3_weights, &output[dataflow_loop], &Layer3_bias);
	}


}


XRef12_3_1 refNN;

int * NN_control = (int *)(0x43C00000);
//float * input_acc = (float *)(0x43C00080); // 4 * 32b
float * output_acc = (float *)(0x43C01000); // 4 * 32b

int main()
{
    init_platform();
    //Xil_DCacheDisable();
    print("Hello World\n\r");

    XTime tStart, tEnd;
	double  ElapsedTime;


    float *Dout_HW = (float *)(0x16000000);
    float *Din_HW = (float *)(0x10000000);

    for(int i = 0; i < 12 * 10; i++){
    	Din_HW[i] = 1;
    }

    int status;
	status = XRef12_3_1_Initialize(&refNN,0);
	if(status != XST_SUCCESS){
	   print("HLS refNN failed\n\r");
	   return -1;
	}

	//XRef12_3_1_Set_inp(&refNN,Din_HW);
	//XRef12_3_1_Set_output_r(&refNN,Dout_HW);
	//XRef12_3_1_EnableAutoRestart(&refNN);


    print("Hello World\n\r");

    // HW Execution
	XTime_GetTime(&tStart);
	NN_control[0] |=0x1;
	while((NN_control[0]&0x2) != 0x2);
	//XRef12_3_1_Start(&refNN);
	//while (!XRef12_3_1_IsReady(&refNN));
	//while (!XRef12_3_1_IsDone(&refNN));
	XTime_GetTime(&tEnd);

	float output[1000];

	ref12_3_1(output);

	int err = 0;
	for(int i = 0; i < dataflow; i++){
		if(abs(output_acc[i] - output[i]) > 0.00001){
			err++;
		}
	}
	printf("Number of error: %d\n", err);

	printf("Output took %llu clock cycles.\n", 2*(tEnd - tStart));
	ElapsedTime = 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000);
	printf(" Output took %.2f us.\n",ElapsedTime);
	printf("  Average execution time for each input %.2f us.\n",ElapsedTime/dataflow);
	//XRef12_3_1_DisableAutoRestart(&refNN);
    cleanup_platform();
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/*float Layer2_weights[40][784];
float Layer3_weights[40][40];
float Layer4_weights[40][40];
float Layer5_weights[10][40];
float Layer2_bias[40];
float Layer3_bias[40];
float Layer4_bias[40];
float Layer5_bias[10];*/

#include "Layer2_weights.h"
#include "Layer3_weights.h"
#include "Layer4_weights.h"
#include "Layer5_weights.h"
#include "Layer2_bias.h"
#include "Layer3_bias.h"
#include "Layer4_bias.h"
#include "Layer5_bias.h"

void stage0(float inp_local[784], float Layer2_weights[40][784], float Layer2_bias[40], float Layer2_neurons[40]){
	float sum;
	for(int i = 0 ; i < 40 ; i++){
		sum = 0;
		for(int j = 0 ; j < 784 ; j++)
			sum += inp_local[j] * Layer2_weights[i][j];
		Layer2_neurons[i] = (sum + Layer2_bias[i]) > 0 ? (sum + Layer2_bias[i]) : 0;
	}
}

void stage0_opt(float inp_local[784], float Layer2_weights[40][784], float Layer2_bias[40], float Layer2_neurons[40]){
	float sum;
	float sum_p[8] = {0,0,0,0,0,0,0,0};
	for(int i = 0 ; i < 40 ; i++){
		sum = 0;
		for(int j = 0 ; j < 784 ; j+=8){
#pragma HLS PIPELINE II=1
			for(int jj = 0 ; jj < 8 ; jj++)
#pragma HLS UNROLL
				sum_p[jj] = inp_local[j+jj] * Layer2_weights[i][j+jj];
			sum += sum_p[0] + sum_p[1] + sum_p[2] + sum_p[3] + sum_p[4] + sum_p[5] + sum_p[6] + sum_p[7];
		}
		Layer2_neurons[i] = (sum + Layer2_bias[i]) > 0 ? (sum + Layer2_bias[i]) : 0;
	}
}

void stage0_opt2(float inp_local[784], float Layer2_weights[40][784], float Layer2_bias[40], float Layer2_neurons[40]){
	float sum1;
	float sum_p1[8] = {0,0,0,0,0,0,0,0};
	float sum2;
	float sum_p2[8] = {0,0,0,0,0,0,0,0};
	float sum3;
	float sum_p3[8] = {0,0,0,0,0,0,0,0};
	float sum4;
	float sum_p4[8] = {0,0,0,0,0,0,0,0};
	for(int i = 0 ; i < 40 ; i+=4){
		sum1 = 0;
		sum2 = 0;
		sum3 = 0;
		sum4 = 0;
		for(int j = 0 ; j < 784 ; j+=8){
#pragma HLS PIPELINE II=1
			for(int jj = 0 ; jj < 8 ; jj++){
#pragma HLS UNROLL
				sum_p1[jj] = inp_local[j+jj] * Layer2_weights[i][j+jj];
				sum_p2[jj] = inp_local[j+jj] * Layer2_weights[i+1][j+jj];
				sum_p3[jj] = inp_local[j+jj] * Layer2_weights[i+2][j+jj];
				sum_p4[jj] = inp_local[j+jj] * Layer2_weights[i+3][j+jj];
			}
			sum1 += sum_p1[0] + sum_p1[1] + sum_p1[2] + sum_p1[3] + sum_p1[4] + sum_p1[5] + sum_p1[6] + sum_p1[7];
			sum2 += sum_p2[0] + sum_p2[1] + sum_p2[2] + sum_p2[3] + sum_p2[4] + sum_p2[5] + sum_p2[6] + sum_p2[7];
			sum3 += sum_p3[0] + sum_p3[1] + sum_p3[2] + sum_p3[3] + sum_p3[4] + sum_p3[5] + sum_p3[6] + sum_p3[7];
			sum4 += sum_p4[0] + sum_p4[1] + sum_p4[2] + sum_p4[3] + sum_p4[4] + sum_p4[5] + sum_p4[6] + sum_p4[7];
		}
		Layer2_neurons[i] = (sum1 + Layer2_bias[i]) > 0 ? (sum1 + Layer2_bias[i]) : 0;
		Layer2_neurons[i+1] = (sum2 + Layer2_bias[i+1]) > 0 ? (sum2 + Layer2_bias[i+1]) : 0;
		Layer2_neurons[i+2] = (sum3 + Layer2_bias[i+2]) > 0 ? (sum3 + Layer2_bias[i+2]) : 0;
		Layer2_neurons[i+3] = (sum4 + Layer2_bias[i+3]) > 0 ? (sum4 + Layer2_bias[i+3]) : 0;
	}
}

void stage0_opt3(float inp_local[784], float Layer2_weights[40][784], float Layer2_bias[40], float Layer2_neurons[40]){
	float sum1;
	float sum2;
	float sum3;
	float sum4;
	float sum_p[4][8];
#pragma HLS ARRAY_RESHAPE variable=sum_p complete dim=1
#pragma HLS ARRAY_RESHAPE variable=sum_p complete dim=2

	for(int i = 0 ; i < 40 ; i+=4){
		sum1 = 0;
		sum2 = 0;
		sum3 = 0;
		sum4 = 0;
		for(int j = 0 ; j < 784 ; j+=8){
#pragma HLS PIPELINE II=1
			for(int jj = 0 ; jj < 8 ; jj++){
#pragma HLS UNROLL
				sum_p[0][jj] = inp_local[j+jj] * Layer2_weights[i][j+jj];
				sum_p[1][jj] = inp_local[j+jj] * Layer2_weights[i+1][j+jj];
				sum_p[2][jj] = inp_local[j+jj] * Layer2_weights[i+2][j+jj];
				sum_p[3][jj] = inp_local[j+jj] * Layer2_weights[i+3][j+jj];
			}
			sum1 += sum_p[0][0] + sum_p[0][1] + sum_p[0][2] + sum_p[0][3] + sum_p[0][4] + sum_p[0][5] + sum_p[0][6] + sum_p[0][7];
			sum2 += sum_p[1][0] + sum_p[1][1] + sum_p[1][2] + sum_p[1][3] + sum_p[1][4] + sum_p[1][5] + sum_p[1][6] + sum_p[1][7];
			sum3 += sum_p[2][0] + sum_p[2][1] + sum_p[2][2] + sum_p[2][3] + sum_p[2][4] + sum_p[2][5] + sum_p[2][6] + sum_p[2][7];
			sum4 += sum_p[3][0] + sum_p[3][1] + sum_p[3][2] + sum_p[3][3] + sum_p[3][4] + sum_p[3][5] + sum_p[3][6] + sum_p[3][7];
		}
		Layer2_neurons[i] = (sum1 + Layer2_bias[i]) > 0 ? (sum1 + Layer2_bias[i]) : 0;
		Layer2_neurons[i+1] = (sum2 + Layer2_bias[i+1]) > 0 ? (sum2 + Layer2_bias[i+1]) : 0;
		Layer2_neurons[i+2] = (sum3 + Layer2_bias[i+2]) > 0 ? (sum3 + Layer2_bias[i+2]) : 0;
		Layer2_neurons[i+3] = (sum4 + Layer2_bias[i+3]) > 0 ? (sum4 + Layer2_bias[i+3]) : 0;
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
void stage1_opt(float Layer2_neurons[40], float Layer3_weights[40][40], float Layer3_bias[40], float Layer3_neurons[40]){
	float sum;
	float sum_p[4] = {0,0,0,0};
	for(int i = 0 ; i < 40 ; i++){
		sum = 0;
		for(int j = 0 ; j < 40 ; j+=4){
#pragma HLS PIPELINE II=1
			for(int jj = 0 ; jj < 4 ; jj++)
#pragma HLS UNROLL
				sum_p[jj] = Layer2_neurons[j+jj] * Layer3_weights[i][j+jj];
			sum += sum_p[0] + sum_p[1] + sum_p[2] + sum_p[3];
		}
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
void stage2_opt(float Layer3_neurons[40], float Layer4_weights[40][40], float Layer4_bias[40], float Layer4_neurons[40]){
	float sum;
	float sum_p[4] = {0,0,0,0};
	for(int i = 0 ; i < 40 ; i++){
		sum = 0;
		for(int j = 0 ; j < 40 ; j+=4){
#pragma HLS PIPELINE II=1
			for(int jj = 0 ; jj < 4 ; jj++)
#pragma HLS UNROLL
				sum_p[jj]= Layer3_neurons[j+jj] * Layer4_weights[i][j+jj];
			sum += sum_p[0] + sum_p[1] + sum_p[2] + sum_p[3];
		}
		Layer4_neurons[i] = (sum + Layer4_bias[i]) > 0 ? (sum + Layer4_bias[i]) : 0;
	}
}
void stage3(float Layer4_neurons[40], float Layer5_weights[10][40], float Layer5_bias[10], float out[10]){
	float sum;
	for(int i = 0 ; i < 10 ; i++){
#pragma HLS PIPELINE II=1
		sum = 0;
		for(int j = 0 ; j < 40 ; j++)
			sum += Layer4_neurons[j] * Layer5_weights[i][j];
		out[i] = sum + Layer5_bias[i];
	}
}

void port2loc(float inp[784], float inp_local[784]){
	for(int init_local = 0; init_local < 784; init_local++){
		inp_local[init_local] = inp[init_local];
	}
}

void ref_top(float inp[784 * 1000] , float out[10 * 1000]){
#pragma HLS INTERFACE m_axi port=inp offset=slave bundle=neuronAXI1
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=neuronAXI2
#pragma HLS INTERFACE s_axilite port=return bundle=control

	float inp_local[784];
	float Layer2_neurons[40];
	float Layer3_neurons[40];
	float Layer4_neurons[40];

	for(int dataflow_loop = 0; dataflow_loop < 1000; dataflow_loop++){
	#pragma HLS DATAFLOW
		port2loc(&inp[dataflow_loop*784], inp_local);
		stage0_opt3(inp_local, Layer2_weights, Layer2_bias, Layer2_neurons);
		stage1_opt(Layer2_neurons, Layer3_weights, Layer3_bias, Layer3_neurons);
		stage2_opt(Layer3_neurons, Layer4_weights, Layer4_bias, Layer4_neurons);
		stage3(Layer4_neurons, Layer5_weights, Layer5_bias, &out[dataflow_loop*10]);
	}

}









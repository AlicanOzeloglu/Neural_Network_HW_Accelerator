
#define dataflow 1000
#include "inp.h"

void stage0(float inp[12], float weight[3][12], float out[3], float bias[3]){
	float sum;
	float sump[4];
#pragma HLS pipeline
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
#pragma HLS pipeline

	for(int j = 0; j < 3; j++){
		sum += inp[j] * weight[j];
	}
	*out = sum + *bias;
}

void localMemLoad(float inp[12], float local_inp[12]){
#pragma HLS pipeline
	for(int i = 0; i < 12; i++){
//#pragma HLS unroll
		local_inp[i] = inp[i];
	}

}

void ref12_3_1(/*float inp[12 * dataflow],*/ float output[dataflow]){
//#pragma HLS INTERFACE s_axilite port=output bundle=neuronAXI1
#pragma HLS INTERFACE s_axilite port=output bundle=neuronAXI
//#pragma HLS INTERFACE m_axi port=output offset=slave bundle=neuronAXI1
//#pragma HLS INTERFACE m_axi port=inp offset=slave bundle=neuronAXI2
#pragma HLS INTERFACE s_axilite port=return bundle=neuronAXI

	float local_inp[12];
#pragma HLS ARRAY_RESHAPE variable=local_inp complete dim=1

	float Layer2_weights[3][12] = {
			{1,1,1,1,1,1,1,1,1,1,1,1},
			{1,1,1,1,1,1,1,1,1,1,1,1},
			{1,1,1,1,1,1,1,1,1,1,1,1}
	};
#pragma HLS ARRAY_RESHAPE variable=Layer2_weights complete dim=1
#pragma HLS ARRAY_RESHAPE variable=Layer2_weights complete dim=2
	float Layer3_weights[3] = {1,1,1};
	float hidden[3];
	float Layer2_bias[3] = {1,1,1};
	float Layer3_bias = 1;

	for(int dataflow_loop = 0; dataflow_loop < dataflow; dataflow_loop++){
#pragma HLS DATAFLOW
		localMemLoad(&inp[12 * dataflow_loop], local_inp);
		stage0(local_inp/*&inp[12 * dataflow_loop]*/, Layer2_weights, hidden, Layer2_bias);
		stage1(hidden, Layer3_weights, &output[dataflow_loop], &Layer3_bias);
	}




}


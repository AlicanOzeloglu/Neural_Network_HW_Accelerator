# Neural_Network_HW_Accelerator
Fully-Connected Neural Network Hardware Accelerator


This study contains the necessary source codes for the hardware accelerator of the fully-connected neural network model with 784-40-40-40-10 neuron architecture. The design has a structure suitable for the dataflow method. The activation function is "relu". The RTL design was created using a high level synthesizer (Vivado HLS 2017.4).

######  Creating Neural Network IP with HLS: ######
1. Open Vivado HLS 2017.4
2. Create New Project
3. Right click on Source and click add files.
4. Add HLS/core.cpp to the project as source file.
5. Click C synthesis.
6. When synthesis is finished, click Export RTL.

######  Creating SoC Design:  ######
1. Open Vivado 2017.4
2. Create RTL Project
3. Choose destination board
4. Create block design
5. Click Window --> Ip Catalog --> right click --> Add Repository --> choose the path of HLS file that you create for neural network ip.
6. Design the block design exact same with Vivado/design_1.pdf.
7. Right click on design file and click Create HDL Wrapper.
8. Right click on design and click Generate Output Products.
9. Click on Generate Bitstream.


When generate bitstream finished, export hardware (include bitstream should be marked) and launch SDK.

1. Create new Hello World application project in SDK.
2. Replace the contents of the Hello world project with SDK/main.c
3. Ready to run.

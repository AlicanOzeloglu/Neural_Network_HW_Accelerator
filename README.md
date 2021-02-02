# Deep Reinforcement Learning-based Autonomous Parking Design with Neural Network Compute Accelerators

## Abstract: 

Partially autonomous driving features in vehicles have become very popular today. Machine learning approaches provide successful results in these kind of tasks towards fully autonomous driving. In this study, our main objective is to come up with an autonomous prototype vehicle which finds a parking slot and park itself via deep reinforcement learning algorithm using a neural network. To perform an autonomous parking for our prototype vehicle, two different artificial neural networks (ANN) are trained using deep reinforcement learning (RL) algorithm and embedded into the computing platform of the prototype car. One of the ANNs enables the vehicle to drive autonomously in the parking environment. At the same time, the image processing algorithm is used to determine the emptiness of the parking lot. When the image processing algorithm finds the suitable parking lot, a different ANN is activated and performs the safe parking process. However, neural network (NN)-based machine learning techniques have a high computational burden and require high processing power. Graphics processing unit (GPU) based solutions suffers from its power hungry nature and often not viable for use in real car. Application-specific hardware design brings higher performance with less power consumption since all the logic resources are fully exploited and connected according to the algorithm of interest which makes it an energy efficient solution. High-performance and energy efficient hardwar e accelerators for ANNs are designed and generated via Vivado high-level synthesis (HLS) tool and trained in a simulation environment. The artificial neural network accelerators, sensor and camera interface RTL designs and their software part on Zynq-ARM processor are implemented on the ZedBoard development board with a HW/SW design methodology. NN models for autonomous parking problem are accelerated 17 times compared to the ARM software implementation. For deeper fully-connected layers used in deep RL-based solutions, function-level parallelism (dataflow) is employed to improve the computational efficiency. Our proposed stage-level description for fully connected layers outperforms recent studies in terms ofcomputation time.

## Neural_Network_HW_Accelerator
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




## Publication

- A. Özeloğlu and İ. San, "Acceleration of Neural Network Training on Hardware via HLS for an Edge-AI Device," 2020 Innovations in Intelligent Systems and Applications Conference (ASYU), Istanbul, Turkey, 2020, pp. 1-6, Oct 2020. [Paper](https://ieeexplore.ieee.org/abstract/document/9259845)

- A. Özeloğlu, İ. G. Gürbüz and İ. San, "Deep Reinforcement Learning-based Autonomous Parking Design
with Neural Network Compute Accelerators," submitted, Feb 2021. [Paper](under review)

## Contact

Alican Özeloğlu <br />
Department of Electrical and Electronics Engineering <br />
alican.ozeloglu@tubitak.gov.tr <br />

İsmihan Gül Gürbüz <br />
Department of Electrical and Electronics Engineering <br />
gul.gurbuz@tubitak.gov.tr <br />

İsmail San <br />
Department of Electrical and Electronics Engineering <br />
Eskisehir Technial University <br />
isan83@gmail.com <br />
https://akademik.eskisehir.edu.tr/isan <br />

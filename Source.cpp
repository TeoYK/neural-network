#pragma once


#include <iostream>
#include <fstream>
#include <sstream>

// ---------------------------- setting variables --------------------

constexpr float learningRate = 0.001f;
constexpr int inputNodes = 784;
constexpr int hiddenNodes = 128;
constexpr int outputNodes = 10;
constexpr int dataSize = 60000;
constexpr int testDataSize = 10000;
constexpr int numEpochs = 10;

int label[dataSize];
std::string filename = "mnist.csv";

// ----------------------------------- layer 1 ---------------------------

float inputData[dataSize][inputNodes];
float inputArray[inputNodes];             // get from dataset
float weightsL1[inputNodes][hiddenNodes]; // init at random
float biasesL1[hiddenNodes];              // init at random

float weightedInputL1[hiddenNodes];
float activationL1[hiddenNodes];


// ------------------------------- layer 2 --------------------------------------


// input = activationL1

float weightsL2[hiddenNodes][outputNodes]; // init at random
float biasesL2[outputNodes];               // init at random

float weightedInputL2[outputNodes];


// ---------------------------------------------------------------------------


static void getHiddenLayer() {

	for (int i = 0; i < hiddenNodes; i++) {
		for (int j = 0; j < inputNodes; j++) {
			weightedInputL1[i] += inputArray[j] * weightsL1[j][i]; //normalize input
		}
	}

	for (int i = 0; i < hiddenNodes; i++) {
		weightedInputL1[i] += biasesL1[i];
	}
}



static void getRelu() {

	for (int i = 0; i < hiddenNodes; i++) {

		if (weightedInputL1[i] <= 0.0f) {
			activationL1[i] = 0.0f;
		}
		else {
			activationL1[i] = weightedInputL1[i];
		}
	}
}



static void getOutputLayer() {

	for (int i = 0; i < outputNodes; i++) {
		for (int j = 0; j < hiddenNodes; j++) {
			weightedInputL2[i] += activationL1[j] * weightsL2[j][i];
		}
	}

	for (int i = 0; i < outputNodes; i++) {
		weightedInputL2[i] += biasesL2[i];
	}
}



static float randomFloat() {
	return (float)(rand()) / (float)(RAND_MAX)-0.5f;
}



static void initParameters() {

	srand(time(0u));

	for (int i = 0; i < inputNodes; i++) {
		for (int j = 0; j < hiddenNodes; j++) {
			weightsL1[i][j] = randomFloat(); 	// init weights for layer 1
		}
	}


	for (int i = 0; i < hiddenNodes; i++) {
		biasesL1[i] = randomFloat(); // init biases for layer 1

		for (int j = 0; j < outputNodes; j++) {
			weightsL2[i][j] = randomFloat(); // init weights for layer 2
		}
	}


	for (int i = 0; i < outputNodes;i++) {
		biasesL2[i] = randomFloat(); // init biases for layer 2
	}

}


static void resetGradients() {
	for (int i = 0;i < hiddenNodes;i++) {
		weightedInputL1[i] = 0;
	}

	for (int i = 0;i < outputNodes;i++) {
		weightedInputL2[i] = 0;
	}
}




static void readMNISTRow(int label[]) {

	std::ifstream file(filename);
	std::string line;


	int dataCounter = 0;
	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string value;

		std::getline(ss, value, ',');
		label[dataCounter] = std::stoi(value);

		int pixelIndex = 0;
		while (std::getline(ss, value, ',') && pixelIndex < 784) {
			inputData[dataCounter][pixelIndex] = std::stof(value) / 255.0f;
			pixelIndex++;
		}

		dataCounter++;

		if (dataCounter == dataSize) {
			break;
		}
	}
	file.close();
}


static void forwardPass() {

	resetGradients();

	getHiddenLayer();   // calculate value for weightedInputL1
	getRelu();          // calculate value for activationL1

	getOutputLayer();   // calculate value for weightedInputL2

}


static float backProp(float target[outputNodes]) {
	
	float dLdW2[hiddenNodes][outputNodes];
	float dLdB2[outputNodes];

	float dLdW1[inputNodes][hiddenNodes];
	float dLdB1[hiddenNodes];
	
	// ---------------- compute dLdW2 -----------------------
	

	// compute dLdWI2 = -2 (y-WI2)
	float dLdWI2[outputNodes] = { 0 };
	for (int i = 0; i < outputNodes; i++) {
		dLdWI2[i] = -2.0f * (target[i] - weightedInputL2[i]);
	}

	// dLdW2 = dLdWI2 * dWI2dW2  
	for (int i = 0; i < hiddenNodes; i++) {
		for (int j = 0; j < outputNodes; j++) {
			dLdW2[i][j] = dLdWI2[j] * activationL1[i];
		}
	}

	// -------------- compute dLdB2 -------------------------

	// dLdB2 = dLdWI2 * dWI2dB2 
	for (int i = 0; i < outputNodes; i++) {
		dLdB2[i] = dLdWI2[i] * 1.0f;
	}

	// ---------------- compute dLdW1 ------------------------


	// dLdA1 = dLdWI2 * dWI2dA1
	float dLdA1[hiddenNodes] = { 0 };
	for (int i = 0; i < hiddenNodes; i++) {
		for (int j = 0; j < outputNodes;j++) {
			dLdA1[i] += dLdWI2[j] * weightsL2[i][j];
		}
	}

	// dLdW1 = dLdWI2 * dWI2dA1 * dA1dWI1 * dWI1dW1   
	for (int i = 0; i < inputNodes; i++) {
		for (int j = 0; j < hiddenNodes; j++) {
			dLdW1[i][j] = dLdA1[j] * ((weightedInputL1[j] > 0) ? 1.0f : 0.0f) * inputArray[i];
		}
	}


	// --------------------- compute dLdB1 ------------------------
	
	// dLdW1 = dLdWI2 * dWI2dA1 * dA1dWI1 * dWI1dB1 

	for (int i = 0; i < hiddenNodes; i++) {
		dLdB1[i] = dLdA1[i] * ((weightedInputL1[i] > 0) ? 1.0f : 0.0f) * 1.0f;
	}

	// ------------------------ update parameters --------------



	// ------ update dW2 -------------

	for (int i = 0; i < hiddenNodes; i++) {
		for (int j = 0; j < outputNodes; j++) {
			weightsL2[i][j] -= learningRate * dLdW2[i][j];
		}
	}


	// ------ update dB2 -------------

	for (int i = 0; i < outputNodes; i++) {
		biasesL2[i] -= learningRate * dLdB2[i];
	}



	// ------ update dW1 -------------

	for (int i = 0; i < inputNodes; i++) {
		for (int j = 0; j < hiddenNodes; j++) {
			weightsL1[i][j] -= learningRate * dLdW1[i][j];
		}
	}


	// ------ update dB1 -------------

	for (int i = 0; i < hiddenNodes; i++) {
		biasesL1[i] -= learningRate * dLdB1[i];
	}

	// ------------------ return total loss-------------------------------

	float totalLoss = 0;
	for (int i = 0;i < outputNodes;i++) {
		totalLoss += ((target[i] - weightedInputL2[i]) * (target[i] - weightedInputL2[i]));
	}

	return totalLoss;
	
}



int main() {
	// ------------------------------- main training loop ------------------------------------
	initParameters();
	
	std::cout << "getting data..." << std::endl;
	readMNISTRow(label);
	std::cout << "data loaded" << std::endl;


	for (int epoch = 0; epoch < numEpochs; epoch++) {
		float totalLoss = 0;

		for (int i = 0; i < dataSize - testDataSize; i++) {
		
			float target[outputNodes] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
			target[label[i]] = 1.0f;  // set target array

			for (int j = 0; j < inputNodes;j++) {
				inputArray[j] = inputData[i][j];    // get input array
			}

			forwardPass();
			
			float loss = backProp(target);
			totalLoss += loss;

		}

		std::cout << "epoch : " << epoch + 1 << " loss : " << totalLoss / (dataSize - testDataSize) << std::endl;
	}



	// -------------------------------------- for testing use -------------------------------------


	float correctCounter = 0;
	float allCounter = 0;

	for (int i = dataSize-testDataSize; i < dataSize; i++) {

		for (int j = 0; j < inputNodes;j++) {
			inputArray[j] = inputData[i][j];
		} // get input array

		forwardPass();

		float maxValue = -3.4028235e+38f;
		int pred = 0;
		for (int j = 0;j < outputNodes;j++) {
			if (weightedInputL2[j] > maxValue) {
				maxValue = weightedInputL2[j];
				pred = j;
			}
		}

		if (label[i] == pred) {
			correctCounter++;
		}
		allCounter++;

		std::cout << "label: " << label[i] << ", pred: " << pred << " correct: " << correctCounter*100/allCounter <<"%" << std::endl;
	}


	
	

}
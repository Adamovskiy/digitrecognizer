package info.adamovskiy.nn;

import info.adamovskiy.nn.neuron.InputNeuron;
import info.adamovskiy.nn.neuron.OutputNeuron;

public class LeastSquaresNeuralNetwork extends SimpleNeuralNetwork {
	public LeastSquaresNeuralNetwork(double learningRate, InputNeuron[] inputs,	InputNeuron[] constantInputs, OutputNeuron[] outputs) {
		super(learningRate, inputs, constantInputs, outputs);
	}

	@Override
	public double errorDerivative(double outputValue, double etalonValue) {
		return outputValue - etalonValue;
	}
	
	@Override
	public double error() {
		double sum = 0d;
		for (OutputNeuron output : getOutputs()) {
			double dif = output.getOutputValue() - output.getEtalonValue();
			sum += dif*dif;
		}
		return sum / 2d;
	}
}

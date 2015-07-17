package info.adamovskiy.nn.neuron;

import info.adamovskiy.nn.NeuralNetwork;

public interface Neuron extends NeuralNode {
	
	public void addInput(NeuralNode input, double weight);

	public double getInputWeight(Neuron neuron);

	public double getErrorDerivativeBySum(NeuralNetwork host);
}

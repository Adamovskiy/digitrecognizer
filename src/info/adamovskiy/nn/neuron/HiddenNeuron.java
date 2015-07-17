package info.adamovskiy.nn.neuron;

import info.adamovskiy.nn.NeuralNetwork;

public abstract class HiddenNeuron extends AbstractNeuron {
	public HiddenNeuron(Object label) {
		super(label);
	}

	@Override
	public double calculateErrorDerivativeBySum(NeuralNetwork host) {
		double result = 0;
		for (int i = 0; i < outputs.size(); i++) {
			final Neuron output = outputs.get(i);
			result += getOutputWeight(i) * output.getErrorDerivativeBySum(host);
		}
		return getActivationDerivativeOfSum() * result;
	}
}

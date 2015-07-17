package info.adamovskiy.nn.neuron;

import info.adamovskiy.nn.NeuralNetwork;
import info.adamovskiy.nn.NeuralNetwork.WeightChangedListener;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public abstract class AbstractNeuron extends AbstractNeuralNode implements Neuron {
	// TODO throw ConcurrentModificationException when underlying neuron is changed
	private class InputIterator implements NeuronIterator {
		private int cursor = -1;

		private void checkValidity() {
			if (cursor == -1)
				throw new IllegalStateException();
		}

		@Override
		public double getWeight() {
			checkValidity();
			return weights.get(cursor);
		}

		@Override
		public NeuralNode getNeuron() {
			checkValidity();
			return inputs.get(cursor);
		}

		@Override
		public boolean stepNext() {
			checkValidity();
			if (++cursor == inputs.size()) {
				cursor = -1;
				return false;
			}
			return true;
		}

		@Override
		public boolean moveToFirst() {
			if (inputs.isEmpty())
				return false;
			cursor = 0;
			return true;
		}
	}
	protected final List<NeuralNode> inputs;
	protected final List<Double> weights;
	
	private Double sum = null;
	private Double activationDerivativeOfSum = null;
	private Double errorDerivativeBySum = null;
	
	public AbstractNeuron(Object label) {
		super(label);
		inputs = new ArrayList<>();
		weights = new ArrayList<>();
	}
	
	private double getSum() {
		if (sum != null)
			return sum;
		double scalarSum = 0;
		for (int i = 0; i < inputs.size(); i++) {
			scalarSum += inputs.get(i).getOutputValue() * weights.get(i);
		}
		sum = scalarSum;
		return scalarSum;
	}
	
	protected abstract double activationDerivative(double x);
	
	protected abstract double activation(double x);
	
	protected abstract double calculateErrorDerivativeBySum(NeuralNetwork host);
	
	protected double getOutputWeight(int outputIndex) {
		return outputs.get(outputIndex).getInputWeight(this);
	}
	
	protected double getActivationDerivativeOfSum() {
		if (activationDerivativeOfSum == null) {
			activationDerivativeOfSum = activationDerivative(getSum());
		}
		return activationDerivativeOfSum;
	}
	
	@Override
	public void addInput(NeuralNode input, double weight) {
		inputs.add(input);
		input.addOutput(this);
		weights.add(weight);
	}
	
	@Override
	public double getOutputValue() {
		return activation(getSum());
	}
	
	@Override
	public double backpropagation(NeuralNetwork host, WeightChangedListener weightChangedListener) {
		double effectSum = 0;
		for (int i = 0; i < weights.size(); i++) {
			final double newWeight = weights.get(i) - host.getLearningRate() * inputs.get(i).getOutputValue() * getErrorDerivativeBySum(host);
			effectSum += Math.abs(weights.get(i) - newWeight);
			if (weightChangedListener != null)
				weightChangedListener.onWeightChanged(inputs.get(i), this, weights.get(i), newWeight);
			weights.set(i, newWeight);
		}
		for (int i = 0; i < inputs.size(); i++) {
			effectSum += inputs.get(i).backpropagation(host, weightChangedListener);
		}
		return effectSum;
	}
	
	@Override
	public double getInputWeight(Neuron input) {
		final int idx = inputs.indexOf(input);
		if (idx == -1)
			throw new IllegalArgumentException("Given neuron is not input");
		return weights.get(idx);
	}
	
	@Override
	public double getErrorDerivativeBySum(NeuralNetwork host) {
		if (errorDerivativeBySum == null)
			errorDerivativeBySum = calculateErrorDerivativeBySum(host);
		return errorDerivativeBySum;
	}
	
	@Override
	public void erase() {
		sum = null;
		activationDerivativeOfSum = null;
		errorDerivativeBySum = null;
		super.erase();
	}
	
	@Override
	public Set<NeuralNode> getTerminalInputs() {
		Set<NeuralNode> result = new LinkedHashSet<>();
		if (inputs.isEmpty())
			result.add(this);
		else
			for (NeuralNode input : inputs) {
				result.addAll(input.getTerminalInputs());
			}
		return result;
	}

	@Override
	public NeuronIterator getInputIterator() {
		return new InputIterator();
	}
}

package info.adamovskiy.nn;

import info.adamovskiy.nn.neuron.NeuralNode;
import info.adamovskiy.nn.neuron.NeuralNode.NeuronIterator;
import info.adamovskiy.nn.neuron.Neuron;
import info.adamovskiy.nn.neuron.InputNeuron;
import info.adamovskiy.nn.neuron.OutputNeuron;
import info.adamovskiy.nn.utils.NeuralNetworkUtils;

import java.util.LinkedHashSet;
import java.util.Set;

public abstract class SimpleNeuralNetwork implements NeuralNetwork {
	private final InputNeuron[] inputs;
	private final InputNeuron[] constantInputs;
	private final OutputNeuron[] outputs;
	private final double learningRate;
	private final long weightsCount;
	
	private double lastAverageEffect;
	private WeightChangedListener weightChangedListener;
	
	public SimpleNeuralNetwork(double learningRate, InputNeuron[] inputs, InputNeuron[] constantInputs, OutputNeuron[] outputs) {
		this.learningRate = learningRate;
		this.inputs = inputs;
		this.constantInputs = constantInputs;
		this.outputs = outputs;
		weightsCount = calculateWeightsCount();
		boolean assertionEnabled = false;
		assert assertionEnabled = true;
		if (assertionEnabled) { // assert with named exceptions
			checkTopologyValidity();
		}
	}
	
	private long calculateWeightsCount() {
		long[] weightCounter = {0};
		traverseNetwork(new TraversalListener() {
			@Override
			public boolean onEdgeTraversal(double weight, NeuralNode input, NeuralNode output) {
				weightCounter[0]++;
				return true;
			}
		});
		return weightCounter[0];
	}
	
	/**
	 * @param neuron
	 * @param listener
	 * @return true if traversal should be stopped
	 */
	private boolean traverseInputs(NeuralNode neuron, TraversalListener listener, Set<NeuralNode> traversedNeurons) {
		if (traversedNeurons.contains(neuron))
			return false;
		traversedNeurons.add(neuron);
		NeuronIterator it = neuron.getInputIterator();
		if (!it.moveToFirst())
			return false;
		do {
			double weight = it.getWeight();
			final NeuralNode input = it.getNeuron();
			if (!listener.onEdgeTraversal(weight, input, neuron))
				return true;
			if (traverseInputs(input, listener, traversedNeurons))
				return true;
		}
		while (it.stepNext());
		return false;
	}
	
	private void checkTopologyValidity() {
		Set<NeuralNode> terminalInputs = new LinkedHashSet<>();
		for (OutputNeuron output : outputs) {
			terminalInputs.addAll(output.getTerminalInputs());
		}
		int danglingInputs = 0;
		int implicitInputs = terminalInputs.size();
		for (InputNeuron adjustedInput : inputs) {
			if (terminalInputs.contains(adjustedInput))
				implicitInputs--;
			else
				danglingInputs++;
		}
		for (InputNeuron adjustedInput : constantInputs) {
			if (terminalInputs.contains(adjustedInput))
				implicitInputs--;
			else
				danglingInputs++;
		}
		
		String errorCause = "";
		if (implicitInputs > 0)
			errorCause += String.format("%d implicit input(s)", implicitInputs);
		if (danglingInputs > 0) {
			if (!errorCause.isEmpty())
				errorCause += ", ";
			errorCause += String.format("%d dangling input(s)", danglingInputs);
		}
		if (!errorCause.isEmpty())
			throw new IllegalStateException("Network topology is invalid: " + errorCause);
	}
	
	protected OutputNeuron[] getOutputs() {
		return outputs;
	}
	
	@Override
	public void conclude(double[] inputValues) {
		NeuralNetworkUtils.checkVectorParameterSize(inputValues, inputs.length);
		for (int i = 0; i < inputs.length; i++) {
			inputs[i].setInputValue(inputValues[i]);
		}
	}
	
	@Override
	public double[] getResult() {
		double[] results = new double[outputs.length];
		for (int i = 0; i < outputs.length; i++) {
			results[i] = outputs[i].getOutputValue();
		}
		return results;
	}
	
	@Override
	public double getLearningRate() {
		return learningRate;
	}
	
	public double teach(double[] inputValues, double[] etalonValues) {
		NeuralNetworkUtils.checkVectorParameterSize(etalonValues, outputs.length);
		NeuralNetworkUtils.checkVectorParameterSize(inputValues, inputs.length);

		for (int i = 0; i < inputs.length; i++) {
			inputs[i].setInputValue(inputValues[i]);
		}
		
		double effectSum = 0;
		for (int i = 0; i < outputs.length; i++) {
			outputs[i].setEtalonValue(etalonValues[i]);
		}
		for (int i = 0; i < outputs.length; i++) {
			effectSum += outputs[i].backpropagation(this, weightChangedListener);
		}
		if (weightChangedListener != null)
			weightChangedListener.onIterationFinished();
		lastAverageEffect = effectSum / weightsCount;
		return lastAverageEffect;
	}
	
	public double getLastAverageEffect() {
		return lastAverageEffect;
	}
	
	@Override
	public double getError() {
		return error();
	}
	
	@Override
	public double getError(double[] inputValues, double[] etalonValues) {
		NeuralNetworkUtils.checkVectorParameterSize(etalonValues, outputs.length);
		
		conclude(inputValues);
		for (int i = 0; i < outputs.length; i++) {
			outputs[i].setEtalonValue(etalonValues[i]);
		}
		return getError();
	}
	
	@Override
	public void erase() {
		for (InputNeuron input : inputs) {
			input.erase();
		}
	}
	
	@Override
	public void traverseNetwork(TraversalListener listener) {
		Set<NeuralNode> traversedNeurons = new LinkedHashSet<>();
		for (Neuron output : outputs) {
			if (traverseInputs(output, listener, traversedNeurons))
				return;
		}
	}
	
	@Override
	public void setWeightChangedListener(WeightChangedListener listener) {
		this.weightChangedListener = listener;
	}
	
	@Override
	public long getWeightsCount() {
		return weightsCount;
	}
}

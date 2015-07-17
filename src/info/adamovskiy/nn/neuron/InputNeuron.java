package info.adamovskiy.nn.neuron;

import info.adamovskiy.nn.NeuralNetwork;
import info.adamovskiy.nn.NeuralNetwork.WeightChangedListener;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;

public class InputNeuron extends AbstractNeuralNode {
	private Double inputValue = null;
	
	public InputNeuron(Object label) {
		super(label);
	}

	@Override
	public double getOutputValue() {
		if (inputValue == null)
			throw new IllegalStateException("input value was not set");
		return inputValue;
	}

	public void setInputValue(double inputValue) {
		this.inputValue = inputValue;
	}
	
	@Override
	public double backpropagation(NeuralNetwork host, WeightChangedListener weightChangedListener) {
		return 0;
	}
	
	@Override
	public void erase() {
		inputValue = null;
		
		for (Neuron output : outputs) {
			output.erase();
		}
	}
	
	@Override
	public Set<NeuralNode> getTerminalInputs() {
		return new LinkedHashSet<NeuralNode>(Arrays.asList(this));
	}
	
	//for debug
	@Override
	public String toString() {
		return inputValue == null ? "" : inputValue.toString();
	}
	
	@Override
	public NeuronIterator getInputIterator() {
		return EmptyNeuronIterator.getInstance();
	}
}

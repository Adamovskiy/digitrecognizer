package info.adamovskiy.nn;

import info.adamovskiy.nn.neuron.HiddenNeuron;
import info.adamovskiy.nn.neuron.InputNeuron;
import info.adamovskiy.nn.neuron.OutputNeuron;

public interface NeuronBuilder {
	public HiddenNeuron buildHiddenNeuron(Object label);
	public InputNeuron buildInput(Object label);
	public InputNeuron buildShiftNeuron(Object label);
	public OutputNeuron buildOutput(Object label);
}

package info.adamovskiy.nn;

import info.adamovskiy.nn.neuron.NeuralNode;
import info.adamovskiy.nn.neuron.Neuron;
import info.adamovskiy.nn.neuron.InputNeuron;
import info.adamovskiy.nn.neuron.OutputNeuron;
import info.adamovskiy.nn.neuron.SigmoidNeuronBuilder;

import java.util.Random;

public class NeuralNetworkBuilder {
	public static class LayeredNeuronLabel {
		public final int layer;
		public final int position;
		
		public LayeredNeuronLabel(int layer, int position) {
			this.layer = layer;
			this.position = position;
		}
		
		@Override
		public String toString() {
			return String.format("(%d,%d)", layer, position);
		}
	}
	
	private static final Random rnd = new Random(1);
	
	public static NeuralNetwork createSigmoidPreceptron(double learningRate, double sigmoidAlpha, int inputs, int outputs, Integer... hiddenLayers) {
		assert(hiddenLayers.length >= 0);
		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();
		builder.setNeuronBuilder(new SigmoidNeuronBuilder(sigmoidAlpha));
		builder.addInputLayer(inputs);
		for (int i = 0; i < hiddenLayers.length; ++i) {
			builder.addHiddenLayer(hiddenLayers[i]);
		}
		builder.addOutputLayer(outputs);
		builder.setLearningRate(learningRate);
		return builder.build();
	}

	private InputNeuron[] inputs;
	private OutputNeuron[] outputs;
	private Neuron[] lastBuiltLayer;
	private int layersCounter;
	private InputNeuron shiftNeuron;
	private NeuronBuilder neuronBuilder;
	private double learningRate;
	
	private NeuralNetworkBuilder() {
	}
	
	private void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	private void setNeuronBuilder(NeuronBuilder neuronBuilder) {
		this.neuronBuilder = neuronBuilder;
		shiftNeuron = neuronBuilder.buildShiftNeuron(new LayeredNeuronLabel(-1, -1));
	}
	
	private void addOutputLayer(int outputsCount) {
		if (inputs == null)
			throw new IllegalStateException("Input layer must be set first");
		
		OutputNeuron[] currentLayer = new OutputNeuron[outputsCount];
		NeuralNode[] previousLayer = lastBuiltLayer == null ? inputs : lastBuiltLayer;
		
		layersCounter++;
		for (int i = 0; i < outputsCount; i++) {
			currentLayer[i] = neuronBuilder.buildOutput(new LayeredNeuronLabel(layersCounter, i));
			currentLayer[i].addInput(shiftNeuron, rnd.nextDouble() * 2 - 1);
			for (int j = 0; j < previousLayer.length; j++) {
				currentLayer[i].addInput(previousLayer[j], rnd.nextDouble() * 2 - 1);
			}
		}
		outputs = currentLayer;
	}

	private void addHiddenLayer(int hiddenNeuronsCount) {
		if (inputs == null)
			throw new IllegalStateException("Input layer must be set first");
		if (outputs != null)
			throw new IllegalStateException("Hidden layer can not be added after output layer");
		
		Neuron[] currentLayer = new Neuron[hiddenNeuronsCount];
		NeuralNode[] previousLayer = (lastBuiltLayer == null ? inputs : lastBuiltLayer);
		layersCounter++;
		for (int i = 0; i < hiddenNeuronsCount; i++) {
			currentLayer[i] = neuronBuilder.buildHiddenNeuron(new LayeredNeuronLabel(layersCounter, i));
			currentLayer[i].addInput(shiftNeuron, rnd.nextDouble() * 2 - 1);
			for (int j = 0; j < previousLayer.length; j++) {
				currentLayer[i].addInput(previousLayer[j], rnd.nextDouble() * 2 - 1);
			}
		}
		lastBuiltLayer = currentLayer;
	}
	
	private void addInputLayer(int inputsCount) {
		if (inputs != null || lastBuiltLayer != null || outputs != null)
			throw new IllegalStateException("Input layer already set");
		inputs = new InputNeuron[inputsCount];
		for (int i = 0; i < inputsCount; i++) {
			inputs[i] = neuronBuilder.buildInput(new LayeredNeuronLabel(0, i));
		}
	}
	
	private NeuralNetwork build() {
		final InputNeuron shiftNeuronAsArray[] = {shiftNeuron};
		return new LeastSquaresNeuralNetwork(learningRate, inputs, shiftNeuronAsArray , outputs);
	}
}

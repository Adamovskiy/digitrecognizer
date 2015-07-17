package info.adamovskiy.nn.neuron;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractNeuralNode implements NeuralNode {
	protected static class EmptyNeuronIterator implements NeuronIterator {
		private EmptyNeuronIterator() {
		}
		
		private static final EmptyNeuronIterator instance = new EmptyNeuronIterator();
		
		public static EmptyNeuronIterator getInstance() {
			return instance;
		}
		
		@Override
		public double getWeight() {
			throw new IllegalStateException();
		}

		@Override
		public Neuron getNeuron() {
			throw new IllegalStateException();
		}

		@Override
		public boolean stepNext() {
			return false;
		}

		@Override
		public boolean moveToFirst() {
			return false;
		}
	}
	
	private final Object label;
	protected final List<Neuron> outputs; // TODO private?
	
	public AbstractNeuralNode(Object label) {
		this.label = label;
		outputs = new ArrayList<>();
	}
	
	@Override
	public Object getLabel() {
		return label;
	}
	
	@Override
	public void addOutput(Neuron output) {
		outputs.add(output);
	}
	
	@Override
	public void erase() {
		for (Neuron output : outputs) {
			output.erase();
		}
	}
}

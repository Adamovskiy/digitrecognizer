package info.adamovskiy.nn.neuron;

import java.util.Set;

import info.adamovskiy.nn.NeuralNetwork;
import info.adamovskiy.nn.NeuralNetwork.WeightChangedListener;

public interface NeuralNode {
	/**
	 * Iterates through some set of neurons. If there was no
	 * {@link #moveToFirst()} call, or its result was false, or last
	 * {@link #stepNext()} returns false - iterator is invalid. Validity can be
	 * returned by {@link #moveToFirst()} call (if underlying set of neurons is
	 * not empty).
	 */
	public interface NeuronIterator {

		/**
		 * @return weight of current neuron.
		 */
		public double getWeight();

		/**
		 * @return current neuron.
		 */
		public NeuralNode getNeuron();

		/**
		 * Moves to next neuron in underlying set. If there is no next neurons -
		 * all subsequent calls of {@link #getWeight()} and {@link #getNeuron()}
		 * will throw {@link IllegalStateException}. Can be fixed by
		 * {@link #moveToFirst()} call. Also {@link IllegalStateException} will
		 * be thrown if there was no first {@link #moveToFirst()} call.
		 * 
		 * @return false if there is no any neurons
		 */
		public boolean stepNext();

		/**
		 * Resets this iterator to first neuron in underlying set. If underlying
		 * set is empty - false will be returned.
		 * 
		 * Must be called to initialize newly created iterator.
		 * 
		 * @return false if underlying set of neurons is empty.
		 */
		public boolean moveToFirst();
	}
	
	// for debug
	public Object getLabel();
	
	/**
	 * For internal use only!
	 * 
	 * @param output
	 */
	void addOutput(Neuron output);
	public double getOutputValue();
	public NeuronIterator getInputIterator();
	// TODO implement getOutputIterator()
	/**
	 * Changes all input weights, calls itself recursively for input neurons
	 * 
	 * @param host
	 * @return effect - sum of weight deltas
	 */
	public double backpropagation(NeuralNetwork host, WeightChangedListener weightChangedListener);
	public void erase();
	
	/**
	 * Can be slow. Recommended for topology correctness check.
	 * @return teminal inputs set.
	 */
	public Set<NeuralNode> getTerminalInputs();
}

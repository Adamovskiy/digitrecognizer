package info.adamovskiy.nn;

import info.adamovskiy.nn.neuron.NeuralNode;
import info.adamovskiy.nn.neuron.Neuron;

public interface NeuralNetwork {
	public void conclude(double[] inputValues);
	
	/**
	 * 
	 * @return average effect
	 */
	public double teach(double[] inputs, double[] etalon);
	
	public double getError();
	
	public double getError(double[] inputValues, double[] etalonValues);
	
	public double getLastAverageEffect();
	
	public double error();
	
	public double errorDerivative(double outputValue, double etalonValue);

	double getLearningRate();
	
	public void erase();

	public interface TraversalListener {
		/**
		 * 
		 * @param weight
		 * @param input
		 * @param output
		 * @return boolean - need to continue.
		 */
		public boolean onEdgeTraversal(double weight, NeuralNode input, NeuralNode output);
	}
	
	public void traverseNetwork(TraversalListener traversalListener);
	
	public interface WeightChangedListener {
		public void onWeightChanged(NeuralNode input, Neuron output, double oldWeight, double newWeight);
		public void onIterationFinished();
	}
	public void setWeightChangedListener(WeightChangedListener listener);
	
	public long getWeightsCount();
	
	public double[] getResult();
}

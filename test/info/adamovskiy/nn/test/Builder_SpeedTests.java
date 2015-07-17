package info.adamovskiy.nn.test;

import java.util.Arrays;

import info.adamovskiy.nn.NeuralNetwork;
import info.adamovskiy.nn.NeuralNetworkBuilder;

public class Builder_SpeedTests extends SpeedTestSuite {
	public static void main(String[] args) {
		new Builder_SpeedTests().launch();
	}
	
	private NeuralNetwork sigmoidPreceptron;
	
	@Override
	protected void initSuite() {
	}

	/*
	 * 1 hidden layer/~400k neurons
	 * ~10M edges
	 */
	@SpeedTest
	public void createSigmoidPreceptron_leastSquares_wide() {
		sigmoidPreceptron = NeuralNetworkBuilder.createSigmoidPreceptron(0.5, 0.5, 15, 10, 384500);
	}
	
	/*
	 * 100 hidden layers/320k neurons
	 * ~10M edges
	 */
	@SpeedTest
	public void createSigmoidPreceptron_leastSquares_deep() {
		Integer[] hiddenLayers = new Integer[100];
		Arrays.fill(hiddenLayers, Integer.valueOf(320));
		sigmoidPreceptron = NeuralNetworkBuilder.createSigmoidPreceptron(0.5, 0.5, 15, 10, hiddenLayers);
		System.out.println(sigmoidPreceptron.getWeightsCount());
	}
	
	/*
	 * 0 hidden layers
	 * ~10M edges
	 */
	@SpeedTest
	public void createSigmoidPreceptron_leastSquares_noHiddenLayersManyInputs() {
		sigmoidPreceptron = NeuralNetworkBuilder.createSigmoidPreceptron(0.5, 0.5, 1000000, 10);
	}
	
	/*
	 * 0 hidden layers
	 * ~10M edges
	 */
	@SpeedTest
	public void createSigmoidPreceptron_leastSquares_noHiddenLayersManyOutputs() {
		sigmoidPreceptron = NeuralNetworkBuilder.createSigmoidPreceptron(0.5, 0.5, 9, 1000000);
	}
}

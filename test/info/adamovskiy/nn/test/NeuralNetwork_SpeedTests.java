package info.adamovskiy.nn.test;

import java.util.Arrays;
import java.util.Random;

import info.adamovskiy.nn.NeuralNetwork;
import info.adamovskiy.nn.NeuralNetworkBuilder;

public class NeuralNetwork_SpeedTests extends SpeedTestSuite {
	public static void main(String[] args) {
		new NeuralNetwork_SpeedTests().launch();
	}
	
	private NeuralNetwork nn;
	private static Random rnd = new Random(1);
	
	private static double[] createRandomVector(int length, double min, double max) {
		double[] result = new double[length];
		for (int i = 0; i < result.length; i++) {
			result[i] = rnd.nextDouble() * (max-min) + min;
		}
		return result;
	}
	
	/*
	 * 1 hidden layer/60k neurons
	 * ~1.5M edges
	 */
	public void createWideNN() {
		nn = NeuralNetworkBuilder.createSigmoidPreceptron(0.5, 0.5, 15, 10, 60000);
	}
	
	/*
	 * 20 hidden layers/2 neurons
	 * 176 edges
	 */
	public void createDeepNN() {
		Integer[] hiddenLayers = new Integer[20];
		Arrays.fill(hiddenLayers, Integer.valueOf(2));
		nn = NeuralNetworkBuilder.createSigmoidPreceptron(0.5, 0.5, 15, 10, hiddenLayers);
	}
	
	/*
	 * 0 hidden layers
	 * ~10M edges
	 */
	public void createNoHiddenLayersManyInputsNN() {
		nn = NeuralNetworkBuilder.createSigmoidPreceptron(0.5, 0.5, 1000000, 10);
	}
	
	/*
	 * 0 hidden layers
	 * ~10M edges
	 */
	public void createNoHiddenLayersManyOutputsNN() {
		nn = NeuralNetworkBuilder.createSigmoidPreceptron(0.5, 0.5, 9, 1000000);
	}
	
	@SpeedTest(initMethod="createWideNN")
	public void teachWide() {
		nn.teach(createRandomVector(15, 0, 1), createRandomVector(10, 0, 1));
	}
	
	@SpeedTest(initMethod="createDeepNN")
	public void teachDeep() {
		nn.teach(createRandomVector(15, 0, 1), createRandomVector(10, 0, 1));
	}
	
	@SpeedTest(initMethod="createNoHiddenLayersManyInputsNN")
	public void teachNoHiddenManyInputs() {
		nn.teach(createRandomVector(1000000, 0, 1), createRandomVector(10, 0, 1));
	}
	
	@SpeedTest(initMethod="createNoHiddenLayersManyOutputsNN")
	public void teachNoHiddenManyOutputs() {
		nn.teach(createRandomVector(9, 0, 1), createRandomVector(1000000, 0, 1));
	}
	
	@SpeedTest(initMethod="createWideNN")
	public void concludeWide() {
		nn.conclude(createRandomVector(15, 0, 1));
	}
	
	@SpeedTest(initMethod="createDeepNN")
	public void concludeDeep() {
		nn.conclude(createRandomVector(15, 0, 1));
	}
	
	@SpeedTest(initMethod="createNoHiddenLayersManyInputsNN")
	public void concludeNoHiddenManyInputs() {
		nn.conclude(createRandomVector(1000000, 0, 1));
	}
	
	@SpeedTest(initMethod="createNoHiddenLayersManyOutputsNN")
	public void concludeNoHiddenManyOutputs() {
		nn.conclude(createRandomVector(9, 0, 1));
	}
}

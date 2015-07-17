package info.adamovskiy.nn.neuron;

import info.adamovskiy.nn.NeuronBuilder;

public class SigmoidNeuronBuilder implements NeuronBuilder {
	private final double alpha;
	
	public static class SigmidHiddenNeuron extends HiddenNeuron{
		private final double alpha;
		
		private SigmidHiddenNeuron(Object label, double alpha) {
			super(label);
			this.alpha = alpha;
		}
		
		@Override
		public double activation(double x) {
			return 1d / (1d + Math.exp(-alpha * x));
		}
		
		@Override
		public double activationDerivative(double x) {
			return alpha * activation(x)* (1d - activation(x));
		}
	}
	
	public static class SigmoidOutputNeuron extends OutputNeuron {
		private final double alpha;
		
		private SigmoidOutputNeuron(Object label, double alpha) {
			super(label);
			this.alpha = alpha;
		}
		
		@Override
		public double activation(double x) {
			return 1d / (1d + Math.exp(-alpha * x));
		}
		
		@Override
		public double activationDerivative(double x) {
			return alpha * activation(x)* (1d - activation(x));
		}
	}
	
	public SigmoidNeuronBuilder(double alpha) {
		this.alpha = alpha;
	}
	
	@Override
	public HiddenNeuron buildHiddenNeuron(Object label) {
		return new SigmidHiddenNeuron(label, alpha);
	}

	@Override
	public InputNeuron buildInput(Object label) {
		return new InputNeuron(label);
	}
	
	@Override
	public InputNeuron buildShiftNeuron(Object label) {
		InputNeuron result = buildInput(label);
		result.setInputValue(1);
		return result;
	}

	@Override
	public OutputNeuron buildOutput(Object label) {
		return new SigmoidOutputNeuron(label, alpha);
	}
}

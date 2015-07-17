package info.adamovskiy.nn.neuron;

import info.adamovskiy.nn.NeuralNetwork;

public abstract class OutputNeuron extends AbstractNeuron {
	private Double etalonValue = null;
	private Double outputValue = null;
	
	public OutputNeuron(Object label) {
		super(label);
	}

	@Override
	public double calculateErrorDerivativeBySum(NeuralNetwork host) {
		return host.errorDerivative(getOutputValue(), getEtalonValue()) * getActivationDerivativeOfSum();
	}

	public void setEtalonValue(double etalonValue) {
		this.etalonValue = etalonValue;
	}
	
	public double getEtalonValue() {
		if (etalonValue == null)
			throw new IllegalStateException("Etalon value was not set");
		return etalonValue;
	}
	
	@Override
	public void erase() {
		etalonValue = null;
		outputValue = null;
		super.erase();
	}
	
	@Override
	public double getOutputValue() {
		if (outputValue == null)
			outputValue = super.getOutputValue();
		return outputValue;
	}

	// for debug
	@Override
	public String toString() {
		String result = "";
		if (etalonValue != null) {
			result = String.format("etalon: %.8f", etalonValue);
			if (outputValue != null)
				result += ", ";
		}
		if (outputValue != null)
			result += String.format("output: %.8f", outputValue);
		return result;
	}
}

package info.adamovskiy.digitrecognizer;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import info.adamovskiy.nn.DataSource;

public class SmallNumbersDataSource implements DataSource {
	private static final double[][] NUMBERS = {
		{0,1,0,
		 1,0,1,
		 1,0,1,
		 1,0,1,
		 0,1,0},
		 
		{0,1,0,
		 1,1,0,
		 0,1,0,
		 0,1,0,
		 1,1,1},
		 
		{0,1,0,
		 1,0,1,
		 0,0,1,
		 0,1,0,
		 1,1,1},
		
		{1,1,0,
		 0,0,1,
		 0,1,1,
		 0,0,1,
		 1,1,0},
		 
		{1,0,1,
		 1,0,1,
		 1,1,1,
		 0,0,1,
		 0,0,1},
		
		{1,1,1,
		 1,0,0,
		 1,1,1,
		 0,0,1,
		 1,1,0},
		 
		{0,1,1,
		 1,0,0,
		 1,1,0,
		 1,0,1,
		 0,1,0},
		 
		{1,1,1,
		 0,0,1,
		 0,1,0,
		 1,0,0,
		 1,0,0},
		 
		{0,1,0,
		 1,0,1,
		 0,1,0,
		 1,0,1,
		 0,1,0},
		 
		{0,1,0,
		 1,0,1,
		 0,1,1,
		 0,0,1,
		 1,1,0}
	};
	
	private static final Random rnd = new Random(1);
	
	private final int noizeLevel;
	
	private double[] input;
	private double[] output;
	
	public SmallNumbersDataSource(int noizeLevel) {
		if (noizeLevel < 0 || noizeLevel > 100)
			throw new IllegalArgumentException("Noize level must be between 0 and 100");
		this.noizeLevel = noizeLevel;
	}
	
	public SmallNumbersDataSource() {
		this(0);
	}
	
	private void addNoize() {
		for (int i = 0; i < input.length; i++) {
			input[i] += rnd.nextGaussian() * (noizeLevel / 100d);
			input[i] = input[i] > 1d ? 1d : input[i] < 0d ? 0d :input[i];
		}
	}

	@Override
	public boolean prepareNext() throws IOException {
		int number = rnd.nextInt(10);
		output = new double[10];
		Arrays.fill(output, 0d);
		output[number] = 1d;
		input = Arrays.copyOf(NUMBERS[number], NUMBERS[number].length);
		addNoize();
		return true;
	}

	@Override
	public double[] getInput() {
		return input;
	}

	@Override
	public double[] getOutput() {
		return output;
	}
}

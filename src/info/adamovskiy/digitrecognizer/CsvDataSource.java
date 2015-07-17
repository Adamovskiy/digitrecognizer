package info.adamovskiy.digitrecognizer;

import info.adamovskiy.nn.DataSource;
import info.adamovskiy.nn.utils.NeuralNetworkUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

/**
 * Required csv format: <li>first line is header (will be skipped) <li>each
 * subsequent line has format <code>o,o,o,i,i,i</code>, where i - input vector
 * components, o - correct output vector components. <li>Size of vectors must be
 * same for each line and set in this datasource's constructor. <li>Correct
 * output vector can be 0 - it means that data is not suitable for learning with
 * teacher.
 *
 */
public abstract class CsvDataSource implements DataSource {
	private final BufferedReader br;
	
	private double[] output;
	private double[] input;
	
	private final int rawInputSize;
	private final int rawOutputSize;
	
	public CsvDataSource(String filename, int rawOutputSize, int rawInputSize) throws IOException {
		br = new BufferedReader(new FileReader(new File(filename)));
		br.readLine(); // skip headers line
		this.rawInputSize = rawInputSize;
		this.rawOutputSize = rawOutputSize;
	}
	
	protected abstract double[] prepareInput(String[] rawValues);

	protected abstract double[] prepareOutput(String[] rawValues);

	@Override
	public boolean prepareNext() throws IOException {
		String newLine = br.readLine();
		if (newLine == null)
			return false;
		String[] columns = newLine.split(",");
		
		NeuralNetworkUtils.checkVectorParameterSize(columns, rawInputSize + rawOutputSize);
		if (rawOutputSize > 0)
			output = prepareOutput(Arrays.copyOf(columns, rawOutputSize));
		input = prepareInput(Arrays.copyOfRange(columns, rawOutputSize, columns.length));
		return true;
	}

	@Override
	public double[] getInput() {
		return input;
	}
	
	@Override
	public double[] getOutput() {
		if (rawOutputSize == 0)
			throw new IllegalStateException("This data source has no correct answers data");
		return output;
	}
}

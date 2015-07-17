package info.adamovskiy.nn;

import java.io.IOException;

/**
 * {@link #getInput()} and {@link #getOutput()} should return same values on every call
 * between {@link #prepareNext()} calls. Multiple calls are expected.
 * Behavior before first {@link #prepareNext()} call is irrelevant.
 * 
 */
public interface DataSource {

	/**
	 * @return false if there is no more data.
	 * @throws IOException
	 */
	public boolean prepareNext() throws IOException;
	
	public double[] getInput();
	
	public double[] getOutput();

}
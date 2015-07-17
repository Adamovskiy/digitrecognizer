package info.adamovskiy.nn.teacher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import info.adamovskiy.nn.DataSource;
import info.adamovskiy.nn.NeuralNetwork;

/**
 * Teach iterations count is set manually.
 */
public class OnlineNeuralTeacher {
	
	private final NeuralNetwork network;
	private DataSource dataSource;
	private List<ResultConsumer> resultConsumers = new ArrayList<>();
	
	private int repeats = 1;
	private long iterations = Long.MAX_VALUE;
	private int currentRepeat = 0;
	private long currentIteration = 0;
	
	public OnlineNeuralTeacher(NeuralNetwork network) {
		this.network = network;
	}
	
	public void setDataSource(DataSource dataSource) {
		this.dataSource = dataSource;
	}
	
	public void addResultConsumer(ResultConsumer resultConsumer) {
		resultConsumers.add(resultConsumer);
	}
	
	public void setRepeats(int repeats) {
		this.repeats = repeats;
	}
	
	/**
	 * Sets how many examples from dataSource will be learned. By default Long.MAX_VALUE.
	 * @param iterations
	 */
	public void setIterations(long iterations) {
		this.iterations = iterations;
	}
	
	public volatile boolean stopped = true;
	
	public void startLearning(int delay) throws IOException {
		startLearning(delay, -1);
	}
	
	public void startLearning(int delay, long steps) throws IOException {
		stopped = false;
		if (dataSource == null)
			throw new IllegalStateException("Data source is not set.");
		learningLoop:
		while (currentIteration < iterations) {
			if (currentRepeat == 0)
				if (!dataSource.prepareNext())
					break;
			while (true) {
				network.erase();
				final double effect = network.teach(dataSource.getInput(), dataSource.getOutput());
				final double error = network.getError();
				if (!resultConsumers.isEmpty()) {
					for (ResultConsumer resultConsumer : resultConsumers) {
						resultConsumer.consume(currentIteration, currentRepeat, error, effect, dataSource.getInput(), dataSource.getOutput());
					}
				}
				currentRepeat++;
				if (currentRepeat == repeats) {
					currentRepeat = 0;
					currentIteration++;
				}
				if (stopped || --steps == 0)
					break learningLoop;
				if (delay != 0)
					try {
						Thread.sleep(delay);
					} catch (InterruptedException e) {
						break learningLoop;
					}
				if (currentRepeat == 0)
					break;
			}
		}
		for (ResultConsumer resultConsumer : resultConsumers) {
			resultConsumer.onLearningStopped();
		}
	}
}

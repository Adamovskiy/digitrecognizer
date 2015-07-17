package info.adamovskiy.nn.teacher;

public abstract class AccumulatingResultConsumer implements ResultConsumer {
	private final int detalization;
	
	private double errorAccumulator;
	private double effectAccumulator;
	private int consumesCounter;
	private long startIteration;
	private int startRepeat;
	private long lastIteration;
	private int lastRepeat;
	
	public AccumulatingResultConsumer(int detalization) {
		if (detalization < 1)
			throw new IllegalArgumentException("Detalization must be > 0");
		this.detalization = detalization;
	}
	
	protected abstract void consumeAccumulated(
			long firstIteration, int firstRepeat,
			long lastIteration, int lastRepeat,
			double averageError, double averageEffect);
	
	@Override
	public void consume(long iteration, int repeat, double error, double effect, final double[] input, final double[] output) {
		// order control
		if (lastIteration > iteration || (lastIteration == iteration && lastRepeat > repeat))
			throw new IllegalArgumentException("Invalid consumptions order");
		lastIteration = iteration;
		lastRepeat = repeat;
		
		if (consumesCounter == 0) {
			startIteration = iteration;
			startRepeat = repeat;
		}
		errorAccumulator += error;
		effectAccumulator += effect;
		consumesCounter++;
		
		if (consumesCounter == detalization) {
			consumeAccumulated(
					startIteration, startRepeat,
					iteration, repeat,
					errorAccumulator / detalization, effectAccumulator / detalization);
			consumesCounter = 0;
			errorAccumulator = 0;
			effectAccumulator = 0;
		}
	}
	
	@Override
	public void onLearningStopped() {
		if (consumesCounter != 0)
			consumeAccumulated(
					startIteration, startRepeat,
					lastIteration, lastRepeat,
					errorAccumulator / consumesCounter, effectAccumulator / consumesCounter);
	}
}
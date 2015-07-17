package info.adamovskiy.nn.teacher;

import java.util.Arrays;

public abstract class MultiAccumulatingResultConsumer implements ResultConsumer {
	private final int detalization;
	
	private final double errorAccumulators[];
	private final double effectAccumulators[];
	private final long bucketVolumes[];
	private int consumesCounter;
	private long startIteration;
	private int startRepeat;
	private long lastIteration;
	private int lastRepeat;
	
	public MultiAccumulatingResultConsumer(int detalization) {
		if (detalization < 1)
			throw new IllegalArgumentException("Detalization must be > 0");
		this.detalization = detalization;
		
		final int bucketsCount = getBucketsCount();
		errorAccumulators = new double[bucketsCount];
		effectAccumulators = new double[bucketsCount];
		bucketVolumes = new long[bucketsCount];
	}
	
	protected abstract int selectBucket(final double input[], final double[] output);
	protected abstract int getBucketsCount();
	
	protected abstract void consumeAccumulated(
			long firstIteration, int firstRepeat,
			long lastIteration, int lastRepeat,
			double[] errorAccumulators, double[] effectAccumulators, long[] bucketVolumes);
	
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
		final int bucket = selectBucket(input, output);
		errorAccumulators[bucket] += error;
		effectAccumulators[bucket] += effect;
		bucketVolumes[bucket]++;
		consumesCounter++;
		
		if (consumesCounter == detalization) {
			consumeAccumulated(
					startIteration, startRepeat,
					iteration, repeat,
					errorAccumulators, effectAccumulators, bucketVolumes);
			consumesCounter = 0;
			Arrays.fill(errorAccumulators, 0);
			Arrays.fill(effectAccumulators, 0);
			Arrays.fill(bucketVolumes, 0);
		}
	}
	
	@Override
	public void onLearningStopped() {
		if (consumesCounter != 0)
			consumeAccumulated(
					startIteration, startRepeat,
					lastIteration, lastRepeat,
					errorAccumulators, effectAccumulators, bucketVolumes);
	}
}
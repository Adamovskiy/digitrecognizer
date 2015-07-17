package info.adamovskiy.nn.teacher;

public interface ResultConsumer {
	public void consume(long iteration, int repeat, double error, double effect, final double[] input, final double[] output);
	public void onLearningStopped();
}
package info.adamovskiy.digitrecognizer;

import java.util.ArrayList;
import java.util.List;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

public class ThroughWeightsDrawer {
	private final int pixelSize;
	private final int inputWidth;
	private final List<Canvas> throughWeightImages;
	
	private final double[][] weights;
	private final double[][] deltas;
	private double maxWeight;
	private double maxDelta;
	private boolean displayDelta = true;
	
	public ThroughWeightsDrawer(int inputWidth, int inputHeight, int pixelSize, int outputsCount) {
		this.pixelSize = pixelSize;
		this.inputWidth = inputWidth;
		this.weights = new double[inputWidth * inputHeight][outputsCount];
		this.deltas = new double[inputWidth * inputHeight][outputsCount];
		throughWeightImages = new ArrayList<>(outputsCount);
		for (int i = 0; i < outputsCount; i++) {
			throughWeightImages.add(new Canvas(inputWidth * pixelSize, inputHeight * pixelSize));
		}
	}
	
	private void drawWeight(int input, int output) {
		final double weight = weights[input][output];
		final double delta = deltas[input][output];
		GraphicsContext currentImageGc = throughWeightImages.get(output).getGraphicsContext2D();
		int h = delta > 0 ? 120 : 0;
		double b = Math.max(0d, Math.min(1d, (weight/maxWeight + 1d) / 2d));
		double s = displayDelta ? Math.max(0d, Math.min(1d, (delta/maxDelta + 1d) / 2d)) : 0d;
		currentImageGc.setFill(Color.hsb(h, s, b));
		currentImageGc.fillRect(pixelSize * (input % inputWidth), pixelSize
				* (input / inputWidth), pixelSize, pixelSize);
	}
	
	private void refreshMaxValues() {
		maxWeight = -1;
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				if (weights[i][j] > maxWeight)
					maxWeight = weights[i][j];
			}
		}
		maxDelta = -1;
		for (int i = 0; i < deltas.length; i++) {
			for (int j = 0; j < deltas[i].length; j++) {
				if (deltas[i][j] > maxDelta)
					maxDelta = deltas[i][j];
			}
		}
	}
	
	private void drawImages() {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				drawWeight(i, j);
			}
		}
	}
	
	public void flushImages() {
		refreshMaxValues();
		drawImages();
	}
	
	public void addWeightChange(int input, int output, double oldWeight, double newWeight) {
		final double delta = newWeight - oldWeight;
		weights[input][output] = newWeight;
		deltas[input][output] = delta;
	}
	
	public void toggleDisplayDelta(boolean displayDelta) {
		this.displayDelta = displayDelta;
		drawImages();
	}
	
	public List<Canvas> getImages() {
		return throughWeightImages;
	}
}

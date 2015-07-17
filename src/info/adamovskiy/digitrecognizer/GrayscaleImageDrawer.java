package info.adamovskiy.digitrecognizer;

import info.adamovskiy.nn.utils.NeuralNetworkUtils;

public abstract class GrayscaleImageDrawer implements ImageDrawer {
	// Can be useful, for example, for border drawing.
//	private final int canvasHeight;
//	private final int canvasWidth;
	private final int inputHeight;
	private final int inputWidth;
	private final int inputPixelSize;
	
	public GrayscaleImageDrawer(int imageHeight, int imageWidth, int inputHeight, int inputWidth) {
		if (imageHeight < inputHeight || imageWidth < inputWidth)
			throw new IllegalArgumentException(String.format(
					"Image is to small: %dx%d. Required minimum size is input size: %dx%d", imageHeight, imageWidth,
					inputHeight, inputWidth));
		this.inputHeight = inputHeight;
		this.inputWidth = inputWidth;
		inputPixelSize = Math.min(imageWidth / inputWidth, imageHeight / inputHeight);
		
//		this.canvasHeight = inputPixelSize * inputHeight;
//		this.canvasWidth = inputPixelSize * inputWidth;
	}
	
	protected abstract void drawPixel(int x, int y, int r, int g, int b);
	
	protected void finishDrawing() {
	}

	@Override
	public void drawNextImage(double[] rawData) {
		NeuralNetworkUtils.checkVectorParameterSize(rawData, inputHeight * inputWidth);
		for (int k = 0; k < rawData.length; k++) {
			int x = k % inputHeight;
			int y = k / inputHeight;
			int grayLevel = (int) (rawData[k]*255);
			for (int i = 0; i < inputPixelSize; i++)
				for (int j = 0; j < inputPixelSize; j++)
					drawPixel(x*inputPixelSize + i, y* inputPixelSize + j, grayLevel, grayLevel, grayLevel);
		}
		finishDrawing();
	}
}

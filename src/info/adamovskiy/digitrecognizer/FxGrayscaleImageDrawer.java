package info.adamovskiy.digitrecognizer;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;

public class FxGrayscaleImageDrawer extends GrayscaleImageDrawer {
	private final WritableImage img;
	private final PixelWriter pw;
	private final GraphicsContext gc;
	private final int imageX;
	private final int imageY;
	
	public FxGrayscaleImageDrawer(int imageWidth, int imageHeight, int inputWidth, int inputHeight, int imageX, int imageY, GraphicsContext gc) {
		super(imageWidth, imageHeight, inputWidth, inputHeight);
		this.gc = gc;
		this.imageX = imageX;
		this.imageY = imageY;
		img = new WritableImage(imageHeight, imageWidth);
		pw = img.getPixelWriter();
	}

	@Override
	protected void drawPixel(int x, int y, int r, int g, int b) {
		pw.setColor(x, y, Color.rgb(r, g, b));
	}
	
	@Override
	protected void finishDrawing() {
		gc.drawImage(img, imageX, imageY);
	}
}

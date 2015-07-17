package info.adamovskiy.digitrecognizer;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;

import info.adamovskiy.nn.DataSource;
import info.adamovskiy.nn.NeuralNetwork;
import info.adamovskiy.nn.NeuralNetwork.WeightChangedListener;
import info.adamovskiy.nn.NeuralNetworkBuilder;
import info.adamovskiy.nn.NeuralNetworkBuilder.LayeredNeuronLabel;
import info.adamovskiy.nn.neuron.NeuralNode;
import info.adamovskiy.nn.neuron.Neuron;
import info.adamovskiy.nn.teacher.AccumulatingResultConsumer;
import info.adamovskiy.nn.teacher.MultiAccumulatingResultConsumer;
import info.adamovskiy.nn.teacher.OnlineNeuralTeacher;
import info.adamovskiy.nn.utils.NeuralNetworkUtils;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.XYChart.Data;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.TextField;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class DigitRecognizerApplication extends Application {
	private static class DigitCsvDataSource extends CsvDataSource {
		public DigitCsvDataSource() throws IOException {
			super("./train.csv", 1, INPUT_SIZE);
		}
		
		@Override
		protected double[] prepareInput(String[] rawValues) {
			double[] result = new double[INPUT_SIZE];
			for (int i = 0; i < rawValues.length; i++) {
				result[i] = Integer.parseInt(rawValues[i]) / 255d;
			}
			return result;
		}
		
		@Override
		protected double[] prepareOutput(String[] rawValues) {
			double[] result = new double[10];
			Arrays.fill(result, 0d);
			result[Integer.parseInt(rawValues[0])] = 1d;
			return result;
		}
	}
	
	//private static final int INPUT_SIZE = 15;
	private static final int INPUT_SIZE = 784;
	private static final Integer[] HIDDEN_LAYERS = {100, 20};
	private static final int OUTPUT_SIZE = 10;
	private static final int REPEATS_COUNT = 1;
	private static final int ITERATIONS_COUNT = 10000;
	private static final int CHART_DETALIZATION = 10;
	private static final double LEARNING_RATE = .1;
	private static final double SIGMOID_ALPHA = .5;
	private static final int DELAY = 0;
	
	private static final int BORDER_SIZE = 5;
	private static final int IMAGE_WIDTH = 300;
	private static final int IMAGE_HEIGHT = 300;
	//private static final int INPUT_WIDTH = 3;
	private static final int INPUT_WIDTH = 28;
	private static final int INPUT_HEIGHT = INPUT_SIZE / INPUT_WIDTH;
	
	//private static final int WEIGHT_PIXEL_SIZE = 20;
	private static final int WEIGHT_PIXEL_SIZE = 4;
	
	private static int findPositiveElement(double[] array) {
		for (int i = 0; i < array.length; i++) {
			if (array[i] > 0)
				return i;
		}
		return -1;
	}
	
	private static LineChart<Number, Number> buildChart(String title) {
		final NumberAxis xAxis = new NumberAxis();
		final NumberAxis yAxis = new NumberAxis();
		xAxis.setLabel("Iterations");
		final LineChart<Number, Number> chart = new LineChart<Number, Number>(xAxis, yAxis);
		chart.setCreateSymbols(false);
		chart.setTitle(title);
		return chart;
	}
	
	private static void saveRegionAsImage(File out, Region region) {
		WritableImage wim = new WritableImage((int)region.getWidth(), (int)region.getHeight());
		region.snapshot(null, wim);
		try {
			ImageIO.write(SwingFXUtils.fromFXImage(wim, null), "png", out);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		launch(args);
	}
	
	private final DataSource dataSource;
	private NeuralNetwork nn;
	private OnlineNeuralTeacher teacher;
	
	private LineChart<Number, Number> errorsChart;
	private LineChart<Number, Number> effectsChart;
	private List<XYChart.Series<Number, Number>> effectSeries;
	private List<XYChart.Series<Number, Number>> errorSeries;
	private LineChart<Number, Number> totalErrorChart;
	private LineChart<Number, Number> totalEffectChart;
	private XYChart.Series<Number, Number> totalEffectSeries;
	private XYChart.Series<Number, Number> totalErrorSeries;
	
	private ImageDrawer inputImageDrawer;
	private Label stepInfoLabel;
	private TextField dumpDotFilename;
	private TextField saveDataDirname;
	private List<ProgressBar> bars;
	
	private Button stepButton;
	private Button processButton;
	private Button pauseButton;
	
	private ThroughWeightsDrawer throughWeightsDrawer;
	private WeightChangedListener throughWeightChangedListener;
	
	public DigitRecognizerApplication() throws IOException {
		//dataSource = new SmallNumbersDataSource(10);
		dataSource = new DigitCsvDataSource();
				
		createNN(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYERS);
		createTeacher();
		createCharts();
		createBars();
	}
	
	private void createNN(int inputs, int outputs, Integer... hiddenLayers) {
		nn = NeuralNetworkBuilder.createSigmoidPreceptron(LEARNING_RATE, SIGMOID_ALPHA, inputs, outputs, hiddenLayers);
	}
	
	private void createTeacher() {
		teacher = new OnlineNeuralTeacher(nn);
		teacher.setDataSource(dataSource);
		
		teacher.addResultConsumer(new MultiAccumulatingResultConsumer(CHART_DETALIZATION) {
			@Override
			protected int selectBucket(double[] input, double[] output) {
				return findPositiveElement(output);
			}
			
			@Override
			protected int getBucketsCount() {
				return OUTPUT_SIZE;
			}
			
			@Override
			protected void consumeAccumulated(long firstIteration, int firstRepeat, long lastIteration, int lastRepeat,
					double[] errorAccumulators, double[] effectAccumulators, long[] bucketVolumes) {
				
				for (int i = 0; i < OUTPUT_SIZE; i++) {
					if (bucketVolumes[i] == 0)
						continue;
					addChartsData(lastIteration * REPEATS_COUNT + lastRepeat, errorAccumulators[i] / bucketVolumes[i],
							effectAccumulators[i] / bucketVolumes[i], i);
				}
			}
		});
		
		teacher.addResultConsumer(new AccumulatingResultConsumer(CHART_DETALIZATION) {
			@Override
			protected void consumeAccumulated(long firstIteration, int firstRepeat, long lastIteration, int lastRepeat,
					double averageError, double averageEffect) {
				addTotalChartData(lastIteration * REPEATS_COUNT + lastRepeat, averageError,	averageEffect);
			}
		});
		
		teacher.setRepeats(REPEATS_COUNT);
		teacher.setIterations(ITERATIONS_COUNT);
	}
	
	private void createCharts() {
		errorsChart = buildChart("Error");
		effectsChart = buildChart("Avg. learning effect");
		totalErrorChart = buildChart("Total error");
		totalEffectChart = buildChart("Total avg. learning effect");
		totalErrorChart.setLegendVisible(false);
		totalEffectChart.setLegendVisible(false);
		totalErrorChart.setPadding(new Insets(0, 0, 30, 0));
		totalEffectChart.setPadding(new Insets(0, 0, 30, 0));
		
		errorSeries = new ArrayList<XYChart.Series<Number, Number>>(OUTPUT_SIZE);
		effectSeries = new ArrayList<XYChart.Series<Number, Number>>(OUTPUT_SIZE);
		for (int i = 0; i < OUTPUT_SIZE; i++) {
			XYChart.Series<Number, Number> errorSery = new XYChart.Series<Number, Number>();
			errorSery.setName(Integer.toString(i));
			errorSeries.add(errorSery);
			errorsChart.getData().add(errorSery);
			XYChart.Series<Number, Number> effectSery = new XYChart.Series<Number, Number>();
			effectSery.setName(Integer.toString(i));
			effectSeries.add(effectSery);
			effectsChart.getData().add(effectSery);
		}
		
		totalErrorSeries = new XYChart.Series<Number, Number>();
		totalEffectSeries = new XYChart.Series<Number, Number>();
		totalErrorChart.getData().add(totalErrorSeries);
		totalEffectChart.getData().add(totalEffectSeries);
	}
	
	private void createBars() {
		bars = new ArrayList<>();
		for (int i = 0; i < OUTPUT_SIZE; i++) {
			bars.add(new ProgressBar(0));
		}
	}
	
	private void createOutputWeightImages() {
		throughWeightsDrawer = new ThroughWeightsDrawer(INPUT_WIDTH, INPUT_HEIGHT, WEIGHT_PIXEL_SIZE, OUTPUT_SIZE);
		
		throughWeightChangedListener = new WeightChangedListener() {
			@Override
			public void onWeightChanged(NeuralNode input, Neuron output, double oldWeight, double newWeight) {
				if (!(input.getLabel() instanceof LayeredNeuronLabel) || !(input.getLabel() instanceof LayeredNeuronLabel))
					throw new IllegalArgumentException("Unknow type of neuron label");
				final LayeredNeuronLabel inputLabel = (LayeredNeuronLabel) input.getLabel();
				final LayeredNeuronLabel outputLabel = (LayeredNeuronLabel) output.getLabel();
				
				if (inputLabel.layer != 0 || outputLabel.layer != HIDDEN_LAYERS.length + 1)
					return;
				throughWeightsDrawer.addWeightChange(inputLabel.position, outputLabel.position, oldWeight, newWeight);
			}
			
			@Override
			public void onIterationFinished() {
				throughWeightsDrawer.flushImages();
			}
		};
	}
	
	private void enableOutputWeightChangeListener(boolean enabled) {
		nn.setWeightChangedListener(enabled ? throughWeightChangedListener : null);
	}
	
	private void addChartsData(long iteration, double error, double effect, int realValue) {
		Platform.runLater(new Runnable() {
			@Override
			public void run() {
				errorSeries.get(realValue).getData().add(new Data<Number, Number>(iteration, error));
				effectSeries.get(realValue).getData().add(new Data<Number, Number>(iteration, effect));
			}
		});
	}
	
	private void addTotalChartData(long iteration, double error, double effect) {
		Platform.runLater(new Runnable() {
			@Override
			public void run() {
				totalErrorSeries.getData().add(new Data<Number, Number>(iteration, error));
				totalEffectSeries.getData().add(new Data<Number, Number>(iteration, effect));
			}
		});
	}
	
	private synchronized void markInProgress(boolean inProgress) {
		pauseButton.setDisable(!inProgress);
		processButton.setDisable(inProgress);
		stepButton.setDisable(inProgress);
	}
	
	private void learnStep() {
		try {
			markInProgress(true);
			long startTime;
			long elapsedTime;
			try {
				enableOutputWeightChangeListener(true);
				startTime = System.currentTimeMillis();
				teacher.startLearning(0, 1);
				elapsedTime = System.currentTimeMillis() - startTime;
				enableOutputWeightChangeListener(false);
			}
			finally {
				markInProgress(false);
			}
			long millis = elapsedTime % 1000;
			long seconds = (elapsedTime / 1000) % 60;
			long minutes = (elapsedTime / (1000 * 60)) % 60;
			long hours = elapsedTime / (1000 * 60 * 60);
		
			stepInfoLabel.setText(
					"Real value: " + findPositiveElement(dataSource.getOutput()) +
					"\nError: " + nn.getError() +
					"\nEffect: " + nn.getLastAverageEffect() +
					"\nIteration duration: " + String.format("%d:%02d:%02d.%04d", hours, minutes, seconds, millis)
					);
			inputImageDrawer.drawNextImage(dataSource.getInput());
			double[] result = nn.getResult();
			for (int i = 0; i < result.length; i++) {
				bars.get(i).setProgress(result[i]);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void startContiniousLearning() {
		final Thread thr = new Thread(new Runnable() {
			@Override
			public void run() {
				try {
					markInProgress(true);
					teacher.startLearning(DELAY);
				} catch (IOException e) {
					e.printStackTrace();
				}
				finally {
					markInProgress(false);
				}
			}
		});
		thr.setDaemon(true);
		thr.start();
	}
	
	private void pauseLearning() {
		teacher.stopped = true;
	}
	
	private void dumpDotFile() {
		try {
			NeuralNetworkUtils.dumpDot(nn, dumpDotFilename.getText());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void saveData() {
		File dir = new File(saveDataDirname.getText());
		dir.mkdir();
		saveRegionAsImage(new File(dir, "total-effect.png"), totalEffectChart);
		saveRegionAsImage(new File(dir, "total-error.png"), totalErrorChart);
		saveRegionAsImage(new File(dir, "effect.png"), effectsChart);
		saveRegionAsImage(new File(dir, "error.png"), errorsChart);
		try(FileWriter writer = new FileWriter(new File(dir, "configs.txt"))) {
			writer.write("INPUT_SIZE: " + INPUT_SIZE);
			writer.write("\nHIDDEN_LAYERS: " + Arrays.toString(HIDDEN_LAYERS));
			writer.write("\nOUTPUT_SIZE: " + OUTPUT_SIZE);
			writer.write("\nREPEATS_COUNT: " + REPEATS_COUNT);
			writer.write("\nITERATIONS_COUNT: " + ITERATIONS_COUNT);
			writer.write("\nCHART_DETALIZATION: " + CHART_DETALIZATION);
			writer.write("\nLEARNING_RATE: " + LEARNING_RATE);
			writer.write("\nSIGMOID_PARAMETER: " + SIGMOID_ALPHA);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void start(Stage primaryStage) throws Exception {
		Canvas canvas = new Canvas(IMAGE_WIDTH + BORDER_SIZE * 2, IMAGE_HEIGHT + BORDER_SIZE * 2);
		final GraphicsContext gc = canvas.getGraphicsContext2D();
		gc.setStroke(Color.BLUE);
		gc.setLineWidth(BORDER_SIZE);
		gc.strokeRect(0, 0, IMAGE_WIDTH + BORDER_SIZE * 2, IMAGE_HEIGHT + BORDER_SIZE * 2);
		
		inputImageDrawer = new FxGrayscaleImageDrawer(IMAGE_WIDTH, IMAGE_HEIGHT, INPUT_WIDTH, INPUT_HEIGHT,
				BORDER_SIZE + 1, BORDER_SIZE + 1, gc);
		
		stepInfoLabel = new Label("Real value: -\nError: -\nEffect: -\nIteration duration: -");
		stepButton = new Button("Learn step");
		
		stepButton.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				learnStep();
			}
		});
		
		dumpDotFilename = new TextField("/tmp/out.dot");
		
		Button dumpDotButton = new Button("Dump DOT");
		dumpDotButton.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				dumpDotFile();
			}
		});
		
		processButton = new Button("Continious learning");
		processButton.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				startContiniousLearning();
			}
		});
		
		pauseButton = new Button("Pause");
		pauseButton.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				pauseLearning();
			}
		});
		pauseButton.setDisable(true);
		
		CheckBox displayDeltas = new CheckBox("Display deltas");
		displayDeltas.setSelected(true);
		displayDeltas.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				throughWeightsDrawer.toggleDisplayDelta(displayDeltas.isSelected());
			}
		});
		
		saveDataDirname = new TextField("/tmp/charts");
		
		Button saveDataButton = new Button("Save data to files");
		saveDataButton.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				saveData();
			}
		});
		
		HBox row1 = new HBox();
		row1.setPadding(new Insets(20));
		VBox column0 = new VBox(20);
		VBox column1 = new VBox(20);
		VBox column2 = new VBox(20);
		VBox column3 = new VBox(20);
		
		for (int i = 0; i < bars.size(); i++) {
			column0.getChildren().add(new Label(Integer.toString(i)));
			column0.getChildren().add(bars.get(i));
		}
		
		VBox infosPanel = new VBox(20);
		infosPanel.setAlignment(Pos.CENTER);
		
		column1.getChildren().add(canvas);
		infosPanel.getChildren().add(stepInfoLabel);
		infosPanel.getChildren().add(stepButton);
		infosPanel.getChildren().add(processButton);
		infosPanel.getChildren().add(pauseButton);
		HBox dumpDotPane = new HBox(new Label("file name:"), dumpDotFilename, dumpDotButton);
		dumpDotPane.setAlignment(Pos.CENTER);
		infosPanel.getChildren().add(dumpDotPane);
		HBox saveDataPane = new HBox(new Label("dir name:"), saveDataDirname, saveDataButton);
		saveDataPane.setAlignment(Pos.CENTER);
		infosPanel.getChildren().add(saveDataPane);
		infosPanel.getChildren().add(displayDeltas);
		column1.getChildren().add(infosPanel);
		
		column2.getChildren().add(errorsChart);
		column2.getChildren().add(effectsChart);
		
		column3.getChildren().add(totalErrorChart);
		column3.getChildren().add(totalEffectChart);
		
		row1.getChildren().add(column1);
		row1.getChildren().add(column0);
		row1.getChildren().add(column2);
		row1.getChildren().add(column3);
		
		HBox row2 = new HBox(20);
		createOutputWeightImages();
		row2.getChildren().addAll(throughWeightsDrawer.getImages());
		
		primaryStage.setScene(new Scene(new VBox(row1, row2), 1500, 1000));
		primaryStage.show();
	}
}

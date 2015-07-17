package info.adamovskiy.nn.utils;

import info.adamovskiy.nn.NeuralNetwork;
import info.adamovskiy.nn.NeuralNetwork.TraversalListener;
import info.adamovskiy.nn.neuron.NeuralNode;
import info.adamovskiy.nn.neuron.InputNeuron;
import info.adamovskiy.nn.neuron.SigmoidNeuronBuilder;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

public abstract class NeuralNetworkUtils {
	@SuppressWarnings("serial")
	private static final Map<Class<? extends NeuralNode>, String> neuronTypeNames = new HashMap<Class<? extends NeuralNode>, String>(){{
		put(InputNeuron.class, "I");
		put(SigmoidNeuronBuilder.SigmidHiddenNeuron.class, "HS");
		put(SigmoidNeuronBuilder.SigmoidOutputNeuron.class, "OS");
	}};
	
	@SuppressWarnings("serial")
	private static final Set<Class<? extends NeuralNode>> labeledNeuronTypes = new LinkedHashSet<Class<? extends NeuralNode>>() {{
		add(InputNeuron.class);
		add(SigmoidNeuronBuilder.SigmoidOutputNeuron.class);
	}};
	
	/**
	 * Dumps network to DOT file. See <a
	 * href="http://www.graphviz.org/content/dot-language">DOT Language
	 * documentation page</a> for details.
	 * 
	 * @param nn
	 *            network to dump
	 * @param filename
	 *            output file
	 * @throws IOException
	 */
	public static void dumpDot(NeuralNetwork nn, String filename) throws IOException {
		try(FileWriter writer = new FileWriter(new File(filename))) {
			writer.write("digraph G {\n");
			IndexedHashSet<NeuralNode> neurons = new IndexedHashSet<>();
			nn.traverseNetwork(new TraversalListener() {
				
				@Override
				public boolean onEdgeTraversal(double weight, NeuralNode input, NeuralNode output) {
					String inputName = neuronTypeNames.get(input.getClass());
					String outputName = neuronTypeNames.get(output.getClass());
					int outputIdx;
					int inputIdx;
					try {
						if (!neurons.contains(output)) {
							outputIdx = neurons.index(output);
							if (labeledNeuronTypes.contains(output.getClass()) && !"".equals(output.toString()))
								writer.write(String.format("%s%d [label=\"%s, %s\"]\n", outputName, outputIdx,
										outputName, output.toString()));
						}
						else
							outputIdx = neurons.getIndex(output);
						if (!neurons.contains(input)) {
							inputIdx = neurons.index(input);
							if (labeledNeuronTypes.contains(input.getClass()) && !"".equals(input.toString()))
								writer.write(String.format("%s%d [label=\"%s, %s\"]\n", inputName, inputIdx, inputName,
										input.toString()));
						}
						else
							inputIdx = neurons.getIndex(input);
					
						writer.write(String.format("%s%d -> %s%d [label=\"%.8f\"]\n", inputName, inputIdx, outputName,
								outputIdx, weight));
					} catch (IOException e) {
						throw new RuntimeException(e);
					}
					return true;
				}
			});
			writer.write("}");
		}
	}
	
	public static void checkVectorParameterSize(double[] param, int expectedSize) {
		if (param.length != expectedSize)
			throw new IllegalArgumentException(String.format("Wrong parameter vector size: expected %d, got %d", expectedSize, param.length));
	}
	
	public static void checkVectorParameterSize(String[] param, int expectedSize) {
		if (param.length != expectedSize)
			throw new IllegalArgumentException(String.format("Wrong parameter vector size: expected %d, got %d", expectedSize, param.length));
	}
}

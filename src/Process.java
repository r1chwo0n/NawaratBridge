import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class Process {
    public static double[][][] randomWeight(int n_in, int[] n_hidden, int n_out) {
        int numHiddenLayers = n_hidden.length;
        double[][][] weights = new double[numHiddenLayers + 1][][];

        weights[0] = new double[n_hidden[0]][n_in + 1];
        initializeWeights(weights[0], n_in);

        for (int layerIndex = 1; layerIndex < numHiddenLayers; layerIndex++) {
            weights[layerIndex] = new double[n_hidden[layerIndex]][n_hidden[layerIndex - 1] + 1];
            initializeWeights(weights[layerIndex], n_hidden[layerIndex - 1]);
        }

        weights[numHiddenLayers] = new double[n_out][n_hidden[numHiddenLayers - 1] + 1];
        initializeWeights(weights[numHiddenLayers], n_hidden[numHiddenLayers - 1]);

        return weights;
    }
    public static double[][][] previousWeight(int n_in, int[] n_hidden, int n_out){
        int numHiddenLayers = n_hidden.length;
        double[][][] weights = new double[numHiddenLayers + 1][][];
        weights[0] = new double[n_hidden[0]][n_in + 1];
        for (int layerIndex = 1; layerIndex < numHiddenLayers; layerIndex++) {
            weights[layerIndex] = new double[n_hidden[layerIndex]][n_hidden[layerIndex - 1] + 1];
        }

        weights[numHiddenLayers] = new double[n_out][n_hidden[numHiddenLayers - 1] + 1];
        return weights;
    }
    private static void initializeWeights(double[][] layerWeights, int numInputs) {
        double min = -1 / Math.sqrt(numInputs + 1);
        double max = 1 / Math.sqrt(numInputs + 1);
        Random random = new Random();

        for (int j = 0; j < layerWeights.length; j++) {
            for (int i = 0; i < numInputs + 1; i++) {
                layerWeights[j][i] = min + (max - min) * random.nextDouble();
            }
        }
    }
    public static void printWeights(double[][][] weights) {
        for (int layerIndex = 0; layerIndex < weights.length; layerIndex++) {
            System.out.println("Layer " + layerIndex + ":");
            for (int nodeIndex = 0; nodeIndex < weights[layerIndex].length; nodeIndex++) {
                System.out.print("Node " + nodeIndex + ": ");
                for (int weightIndex = 0; weightIndex < weights[layerIndex][nodeIndex].length; weightIndex++) {
                    System.out.print(weights[layerIndex][nodeIndex][weightIndex] + " ");
                }
                System.out.println();
            }
            System.out.println();
        }
    }
    public static double[] denormalizeInput(double[] input){
        double[] newInput = new double[input.length];
        int max = 628;
        int min = 95;
        for(int i = 0; i < input.length; i++){
            newInput[i] = (input[i]*(max - min)) + min;
        }
        return newInput;
    }
    public static double[] denormalizeDesired(double[] input){
        double[] newInput = new double[input.length];
        int max = 628;
        int min = 95;
        for(int i = 0; i < input.length; i++){
            newInput[i] = (input[i]*(max - min)) + min;
        }
        return newInput;
    }
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    public static double sigmoidDerivative(double x) {
        return x * (1 - x);
    }
    public static ArrayList<double[]> forward(double[] input, double[][][] weights) {
        ArrayList<double[]> O = new ArrayList<>();
        int numLayers = weights.length; //3
        double[] activations = input; //lenght 3 รวม bias แล้ว

        for (int layerIndex = 0; layerIndex < numLayers; layerIndex++) {
            int numNodes = weights[layerIndex].length;
            int numInputs = weights[layerIndex][0].length;

            double[] newActivations = new double[numNodes]; //ค่า O ของแต่ละ layer

            for (int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++) {
                double sum = 0.0;
                for (int inputIndex = 0; inputIndex < numInputs; inputIndex++) {
                    sum += activations[inputIndex] * weights[layerIndex][nodeIndex][inputIndex];
                }
                newActivations[nodeIndex] = Process.sigmoid(sum);
            }

            double[] withBias = new double[newActivations.length + 1];
            withBias[0] = 1; // Bias term
            System.arraycopy(newActivations, 0, withBias, 1, newActivations.length);
            activations = withBias;
            O.add(activations);
        }
        return O;
    }
    public static double[] calculateOutputError(double[] output, double[] desiredOutput) {
        double[] errors = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            errors[i] = desiredOutput[i] - output[i] ;
        }
        return errors;
    }
    static public double[] gradient(double[] nodeAtOutput,double[] error){ //At output layer
        double[] gradient = new double[nodeAtOutput.length];
        for(int j = 0; j < nodeAtOutput.length; j++){
//            System.out.println("error = "+ error[j] + " diff = " + Process.sigmoidDerivative(nodeAtOutput[j]));
            gradient[j] = error[j] * Process.sigmoidDerivative(nodeAtOutput[j]);
        }
        return gradient;
    }
    static public double[] localGradiant(double[] currentNode, double sumOfGradient){
        double[] local = new double[currentNode.length];
        for(int i = 0; i < currentNode.length; i++){
            local[i] = Process.sigmoidDerivative(currentNode[i]) * sumOfGradient;
        }
        return local;
    }
    public static void printAll(ArrayList<double[]> list) {
        for (double[] array : list) {
            System.out.println(Arrays.toString(array));
        }
    }
    public static void printAllInt(ArrayList<int[]> list) {
        for (int[] array : list) {
            System.out.println(Arrays.toString(array));
        }
    }
    public static double[] getClassIndex(double[] error) {
        if (error[0] > error[1]) {
            return new double[]{0.0,1.0}; // Class 0 1
        } else {
            return new double[]{1.0,0.0}; // Class 1 0
        }
    }

    public static int[] confusionMatrix(ArrayList<double[]> allData,double[][][] weights){
        int[] confusionMatrix = new int[4];

        Collections.shuffle(allData);
        ArrayList<double[]> sample = CrossData.sample(allData, 1);
        ArrayList<double[]> desired_output = CrossData.desired_output(allData);
        for (int k = 0; k < allData.size(); k++) {
            double[] input = sample.get(k); //with bias at index 1
            double[] desiredOutput = desired_output.get(k); //start at index 0, length = 2
            //feed forward get O at output layer
            double[] predictedOutput = Process.forward(input, weights).get(weights.length - 1); // length = 3
            double[] lastOutput = new double[predictedOutput.length - 1]; //length = 2
            for(int i = 1; i < predictedOutput.length; i++){
                lastOutput[i-1] = predictedOutput[i];
            }

            //get error
            double[] getClass = getClassIndex(lastOutput);
            System.out.println("p" + k);
            for(int i = 0; i < getClass.length; i++){
                System.out.println("output = " + getClass[i] + " desired = " + desiredOutput[i]);
            }

            if(desiredOutput[0] == getClass[0]){
                if(desiredOutput[0] == 1) confusionMatrix[0]++; //1 0
                else confusionMatrix[2]++; //0 1
            }else{
                if(desiredOutput[0] == 1) confusionMatrix[1]++; //1 0
                else confusionMatrix[3]++; //0 1
            }
        }
        return confusionMatrix;
    }
}

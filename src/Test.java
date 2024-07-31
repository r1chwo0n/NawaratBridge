import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class Test {
    public static void show(ArrayList<double[]> allData,double[][][] weights){
        Collections.shuffle(allData);

        ArrayList<double[]> sample = FloodData.sample(allData, 1); //norm
        ArrayList<double[]> desired_output = FloodData.desired_output(allData); //norm

        for (int k = 0; k < allData.size(); k++) {
            double[] input = sample.get(k);
            double[] desiredOutput = desired_output.get(k);

            //feed forward
            double[] predictedOutput = Process.forward(input, weights).get(weights.length - 1);
            double[] lastOutput = new double[predictedOutput.length - 1];
            for (int i = 1; i < predictedOutput.length; i++) {
                lastOutput[i - 1] = predictedOutput[i];
            }
            double[] denormOutput = Process.denormalizeInput(lastOutput);
            double[] denormDesired = Process.denormalizeDesired(desiredOutput);
            for(int i = 0; i < denormOutput.length; i++){
                System.out.println("output = " + denormOutput[i] + " desired = " + denormDesired[i]);
            }

        }
    }
    public static void main(String[] args) {
        int n_in = 8;
        int[] n_hidden = {3,4};
        int n_out = 1;
        double learningRate = 0.02;
        double momentumRate = 0.48;
        double[][][] weights = Process.randomWeight(n_in, n_hidden, n_out);

        ArrayList<double[]> allData = FloodData.getAllData(); //not normalize
        Collections.shuffle(allData);

        int folds = 10; // จำนวน fold สำหรับ cross-validation
        int foldSize = allData.size() / folds;

        double totalTrainErrorAv = 0;
        double totalTestErrorAv = 0;

        double sumSquareError = 0;
        double sumTrainErrorAv = 0;
        double sumTestSquareError = 0;
        double sumTestErrorAv = 0;
        double[][][] previousWeight = Process.previousWeight(n_in,n_hidden,n_out);


        ArrayList<double[]> sample = FloodData.sample(allData, 1); //norm
        ArrayList<double[]> desired_output = FloodData.desired_output(allData); //norm

        for (int k = 0; k < 2; k++) { //314

            double[] input = sample.get(k); //with bias at index 1
            double[] desiredOutput = desired_output.get(k); //start at index 0, length = 2

            ArrayList<double[]> forward = Process.forward(input, weights);

            //feed forward get O at output layer
            double[] predictedOutput = forward.get(weights.length - 1); //last layer

            double[] lastOutput = new double[predictedOutput.length - 1];
            for (int i = 1; i < predictedOutput.length; i++) {
                lastOutput[i - 1] = predictedOutput[i];
            }

            //get error
            double[] error = Process.calculateOutputError(lastOutput, desiredOutput);

            //update weight
            double[] gradient = Process.gradient(lastOutput, error);
            double sumOfGradient = 0;

            int layer = weights.length;
            for (int m = layer - 1; m >= 0; m--) { // 0 1
                if (m != 0) {

                    if (m == weights.length - 1) { //last layer
                        for (int j = weights[m].length - 1; j >= 0; j--) { //start at 0
                            for (int i = weights[m - 1].length; i >= 0; i--) {
                                if (k == 0) {
                                    previousWeight[m][j][i] = learningRate * gradient[j] * forward.get(m - 1)[i];
                                } else {
                                    previousWeight[m][j][i] = (momentumRate * previousWeight[m][j][i]) + (learningRate * gradient[j] * (forward.get(m - 1)[i]));
                                }
                                weights[m][j][i] = weights[m][j][i] + previousWeight[m][j][i];
                                sumOfGradient += gradient[j] * weights[m][j][i];
                            }
                        }
//                        System.out.println(sumOfGradient);
                    }else {
//                        System.out.println(sumOfGradient);
                        double temp = sumOfGradient;
                        sumOfGradient = 0;
                        double[] currentNode = new double[Process.forward(input, weights).get(m).length - 1];
                        for (int i = 1; i < Process.forward(input, weights).get(m).length; i++) {
                            currentNode[i - 1] = Process.forward(input, weights).get(m)[i];
                        }
                        for (int j = weights[m].length - 1; j >= 0; j--) {
                            for (int i = weights[m - 1].length; i > 0; i--) {
                                if (k == 0) {
                                    previousWeight[m][j][i] = learningRate * Process.localGradiant(currentNode, temp)[j] * forward.get(m - 1)[i];
                                } else {
                                    previousWeight[m][j][i] = (momentumRate * previousWeight[m][j][i]) + (learningRate * Process.localGradiant(currentNode, temp)[j] * forward.get(m - 1)[i]);
                                }
                                weights[m][j][i] = weights[m][j][i] + previousWeight[m][j][i];
                                sumOfGradient += Process.localGradiant(currentNode, temp)[j] * weights[m][j][i];
                            }
                        }
//                        System.out.println(sumOfGradient);
                    }

                } else {
                    double[] currentNode = new double[Process.forward(input, weights).get(m).length - 1];
                    for (int i = 1; i < Process.forward(input, weights).get(m).length; i++) {
                        currentNode[i - 1] = Process.forward(input, weights).get(m)[i];
                    }
//                    System.out.println(sumOfGradient);
                    for (int j = weights[0].length - 1; j >= 0; j--) {
                        for (int i = input.length - 1; i >= 0; i--) {
                            if (k == 0) {
                                previousWeight[m][j][i] = learningRate * Process.localGradiant(currentNode, sumOfGradient)[j] * input[i];
                            } else {
                                previousWeight[m][j][i] = (momentumRate * previousWeight[m][j][i]) + (learningRate * Process.localGradiant(currentNode, sumOfGradient)[j] * input[i]);
                            }
                            weights[m][j][i] = weights[m][j][i] + previousWeight[m][j][i];
                        }
                    }
                }


            }

        }
    }
}

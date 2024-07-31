import java.util.ArrayList;
import java.util.Collections;
public class Cross {
    public static void main(String[] args) {
        int n_in = 2;
        int[] n_hidden = {15,10};
        int n_out = 2;
        double learningRate = 0.009;
        double momentumRate = 0.165;
        double[][][] weights = Process.randomWeight(n_in, n_hidden, n_out);

        ArrayList<double[]> allData = CrossData.getAllData();
        Collections.shuffle(allData);

        int folds = 10; // จำนวน fold สำหรับ cross-validation
        int foldSize = allData.size() / folds;

        double totalTrainErrorAv = 0;
        double totalTestErrorAv = 0;

        for (int y = 0; y < folds; y++) {
            ArrayList<double[]> testSet = new ArrayList<>(allData.subList(y * foldSize, (y + 1) * foldSize));
            ArrayList<double[]> trainSet = new ArrayList<>(allData);
            trainSet.removeAll(testSet);

            double sumSquareError = 0;
            double sumTrainErrorAv = 0;
            double sumTestSquareError = 0;
            double sumTestErrorAv = 0;

            double[][][] previousWeight = Process.previousWeight(n_in,n_hidden,n_out);
            ArrayList<double[]> sample = CrossData.sample(trainSet, 1);
            ArrayList<double[]> desired_output = CrossData.desired_output(trainSet);

            for (int k = 0; k < trainSet.size(); k++) {
                double[] input = sample.get(k); //with bias at index 1
                double[] desiredOutput = desired_output.get(k); //start at index 0, length = 2
                ArrayList<double[]> forward = Process.forward(input, weights);

                //feed forward get O at output layer
                double[] predictedOutput = forward.get(weights.length - 1); // length = 3
                double[] lastOutput = new double[predictedOutput.length - 1]; //length = 2
                for (int i = 1; i < predictedOutput.length; i++) {
                    lastOutput[i - 1] = predictedOutput[i];
                }

                //get error
                double[] error = Process.calculateOutputError(lastOutput, desiredOutput); // length = 2

                //update weight
                double[] gradient = Process.gradient(lastOutput, error);
                double sumOfGradient = 0;
                //ที่วิ่งเข้า output
                int layer = weights.length; //3
                for (int m = layer - 1; m >= 0; m--) {
                    if (m != 0) {
                        if (m == weights.length - 1) {// layer 2
                            for (int j = weights[m].length - 1; j >= 0; j--) {
                                for (int i = weights[m - 1].length; i >= 0; i--) {
                                    if (k == 0) {
                                        previousWeight[m][j][i] = learningRate * gradient[j] * forward.get(m - 1)[i];
                                    } else {
                                        previousWeight[m][j][i] = (momentumRate * previousWeight[m][j][i]) + (learningRate * gradient[j] * forward.get(m - 1)[i]);
                                    }
                                    weights[m][j][i] = weights[m][j][i] + previousWeight[m][j][i];
                                    sumOfGradient += gradient[j] * weights[m][j][i];
                                }
                            }
                        } else {
                            double temp = sumOfGradient;
                            sumOfGradient = 0;
                            double[] currentNode = new double[forward.get(m).length - 1];
                            for (int i = 1; i < forward.get(m).length; i++) {
                                currentNode[i - 1] = forward.get(m)[i];
                            }
                            for (int j = weights[m].length - 1; j >= 0; j--) {
                                for (int i = weights[m - 1].length; i >= 0; i--) {
                                    if (k == 0) {
                                        previousWeight[m][j][i] = learningRate * Process.localGradiant(currentNode, temp)[j] * forward.get(m - 1)[i];
                                    } else {
                                        previousWeight[m][j][i] = (momentumRate * previousWeight[m][j][i]) + (learningRate * Process.localGradiant(currentNode, temp)[j] * forward.get(m - 1)[i]);
                                    }
                                    weights[m][j][i] = weights[m][j][i] + previousWeight[m][j][i];
                                    sumOfGradient += Process.localGradiant(currentNode, temp)[j] * weights[m][j][i];
                                }
                            }
                        }
                    } else {
                        double[] currentNode = new double[forward.get(m).length - 1];
                        for (int i = 1; i < forward.get(m).length; i++) {
                            currentNode[i - 1] = forward.get(m)[i];
                        }
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
                predictedOutput = Process.forward(input, weights).get(weights.length - 1);
                for (int i = 1; i < predictedOutput.length; i++) {
                    lastOutput[i - 1] = predictedOutput[i];
                }
                error = Process.calculateOutputError(lastOutput, desiredOutput); // length = 2
                double sumError = 0;
                for (int i = 0; i < error.length; i++) {
                    sumError += Math.pow(error[i], 2.0);
                }
                sumSquareError += 0.5 * sumError;
            }
            sumTrainErrorAv = sumSquareError / trainSet.size();
            System.out.println("Fold " + (y + 1) + " - sumTrainErrorAv = " + sumTrainErrorAv);
            totalTrainErrorAv += sumTrainErrorAv;

            sample = CrossData.sample(testSet, 1);
            desired_output = CrossData.desired_output(testSet);

            for(int k = 0; k < testSet.size(); k++){

                double[] input = sample.get(k); //with bias at index 1
                double[] desiredOutput = desired_output.get(k); //start at index 0, length = 2
                double[] predictedOutput = Process.forward(input, weights).get(weights.length - 1); // length = 3
                double[] lastOutput = new double[predictedOutput.length - 1]; //length = 2
                for (int i = 1; i < predictedOutput.length; i++) {
                    lastOutput[i - 1] = predictedOutput[i];
                }
                double[] error = Process.calculateOutputError(lastOutput, desiredOutput);
                double sumTestError = 0;
                for (int i = 0; i < error.length; i++) {
                    sumTestError += Math.pow(error[i], 2.0);
                }
                sumTestSquareError += 0.5 * sumTestError;
            }
            sumTestErrorAv = sumTestSquareError / testSet.size();
            System.out.println("Fold " + (y + 1) + " - sumTestErrorAv = " + sumTestErrorAv);
            totalTestErrorAv += sumTestErrorAv;
        }
        double finalTrainErrorAvg = totalTrainErrorAv / folds;
        double finalTestErrorAvg = totalTestErrorAv / folds;
        System.out.println("Final Average Training Error = " + finalTrainErrorAvg);
        System.out.println("Final Average Testing Error = " + finalTestErrorAvg);

        int[] confuseMatrix = Process.confusionMatrix(allData,weights);
        System.out.println("confuse Matrix");
        System.out.println("number of class a [1,0] T = " + confuseMatrix[0]);
        System.out.println("number of class b [1,0] F = " + confuseMatrix[1]);
        System.out.println("number of class c [0,1] T = " + confuseMatrix[2]);
        System.out.println("number of class d [0,1] F = " + confuseMatrix[3]);

        System.out.println("accuracy = " + (double)(confuseMatrix[0] + confuseMatrix[2])/200);
    }
}

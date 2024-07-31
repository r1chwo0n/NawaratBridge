public class Backward {
    int[] desired_output; //เริ่มเก็บตั้งแต่ 1 เลย
    double[] nodeAtOutput;
    double[] nodeAtHidden;
    double[][] weights;
    int allNode, n_in;
    public Backward(int n_in,int[] desired_output, double[] nodeAtOutput, double[] nodeAtHidden, double[][] weights){
        this.desired_output  = desired_output;
        this.nodeAtOutput = nodeAtOutput;
        this.nodeAtHidden = nodeAtHidden;
        this.weights = weights;
        this.allNode = nodeAtHidden.length + nodeAtOutput.length + 1;
        this.n_in = n_in;
    }
    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }
    private double[] normalization(int[] desired_output){
        double[] normal_out = new double[desired_output.length];
        for(int i = 1; i < desired_output.length; i++){
            normal_out[i] = (double) (desired_output[i] - 95) /(628-95);
        }
        return normal_out;
    }
    public double[] error(double[] nodeAtOutput) { //confuse
        double[] error = new double[desired_output.length];
        for (int i = 1; i < desired_output.length; i++) {
            error[i] = normalization(desired_output)[i] - nodeAtOutput[i];
        }
        return error;
    }
    public double[] gradient(){ //At output layer
        double[] gradient = new double[nodeAtOutput.length];
        for(int j = 1; j < nodeAtOutput.length; j++){
            gradient[j] = error(nodeAtOutput)[j] * sigmoidDerivative(nodeAtOutput[j]);
        }
        return gradient;
    }

    public double sumOfGradient(){
        double sum = 0;
        for(int k = 1; k < nodeAtOutput.length; k++){
            for(int j = 0; j < nodeAtHidden.length ; j++){
                sum += gradient()[k] * weights[k + nodeAtHidden.length-1][j];
            }
        }
        return sum;
    }
    public double[] localGradient(){ //At Hidden Layer
        double[] localGradient = new double[nodeAtHidden.length];
        for(int j = 1; j < nodeAtHidden.length; j++){
            localGradient[j] = sigmoidDerivative(nodeAtHidden[j])*sumOfGradient();
        }
        return localGradient;
    }

}

public class Forward {
    int n_hidden;
    int n_out;
    int[] inputArr;
    int n_in;
    int bias;
    double[][] weights;
    int allNode;
    public Forward(int[] inputArr,int n_hidden,int n_out,int bias,double[][] weights){
        this.n_hidden = n_hidden;
        this.inputArr = inputArr;
        this.n_in = inputArr.length;
        this.bias = bias;
        this.n_out = n_out;
        this.weights = weights;
        this.allNode = n_hidden + n_out + 1;
    }
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    private double[] normalizeInput(int[] input){
        double[] newInput = new double[input.length];
        int max = 628;
        int min = 95;
        for(int i = 0; i < input.length; i++){
            newInput[i] = (double) (input[i] - min) / (max - min);
        }
        return newInput;
    }

    public double[] nodeAtHidden(){
        double[] outputAtHidden = new double[n_hidden+1];
        double[] input = normalizeInput(inputArr);
        for (int l = 1; l < n_hidden + 1; l++){
            double sum = 0;
            int i;
            for (i = 1; i < input.length + 1; i++){
                sum += input[i-1] * this.weights[l][i]; //input start at index 0
            }
            sum += this.weights[l][0]*bias;
            outputAtHidden[l] = sigmoid(sum);
        }
        return outputAtHidden;
    }

    public double[] nodeAtOutput() {
        double[] o_out = new double[n_out+1];
        for(int m = 1; m < n_out+1; m++){
            double sum = 0;
            int i;
            for(i = 1; i < n_hidden+1; i++){
                sum += nodeAtHidden()[i] * this.weights[n_hidden+1][i];
            }
            o_out[m] = sigmoid(sum + this.weights[n_hidden+1][0]*bias);
        }
        return o_out;
    }


}

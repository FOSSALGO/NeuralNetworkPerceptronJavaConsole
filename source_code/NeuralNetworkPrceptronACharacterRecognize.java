package fosalgo;

public class NeuralNetworkPrceptronACharacterRecognize {

    public static void main(String[] args) {
        int[] pola_1 = {1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1};
        int[] pola_2 = {1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1};
        int[] pola_3 = {1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1};
        int[] pola_4 = {1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1};
        int[] pola_5 = {1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1};
        int[] pola_6 = {1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1};

        int[][] input = {pola_1, pola_2, pola_3, pola_4, pola_5, pola_6};
        int[] target = {1, -1, -1, 1, -1, -1};
        double alpha = 1;

        int size = pola_1.length;
        double[] weight = new double[size];//model 

        int epoch = 0;
        boolean training = true;

        while (training) {
            System.out.println("EPOCH-" + (++epoch));
            int sama = 0;
            for (int j = 0; j < input.length; j++) {
                int[] x = input[j];
                int t = target[j];

                //SUMMATION---------------------------
                double net = 0;
                for (int i = 0; i < size; i++) {
                    double xi = x[i];
                    double wi = weight[i];
                    double xiwi = xi * wi;
                    //net = net + xiwi;
                    net += xiwi;
                }

                //ACTIVATION--------------------------
                //theta = 0.5
                //y = output = f(net)
                double theta = 0.5;
                int y = -2;
                if (net > theta) {
                    y = 1;
                } else if (net >= -theta) {
                    y = 0;
                } else {//else if(net<-theta)
                    y = -1;
                }

                //jika y != t maka lakukan update bobot dengan menghitung delta bobot
                double[] deltaWeight = new double[size];
                if (y != t) {
                    for (int i = 0; i < size; i++) {
                        double xi = x[i];
                        double deltaW = alpha * xi * t;
                        deltaWeight[i] = deltaW;
                    }
                } else {
                    sama++;//increment t yang sudah sama dengan y
                }

                //UPDATE BOBOT----------------------------------
                for (int i = 0; i < size; i++) {
                    //weight[i] = weight[i] + deltaWeight[i];
                    weight[i] += deltaWeight[i];
                }
            }

            //kondisi berhenti
            if (sama == target.length) {
                //training = false;
                break;
            }
        }//end of TRAINING

    }
}

package fosalgo;

import java.util.Scanner;

public class NNPerceptronCharRecognizeVis {

    public static void main(String[] args) {
        int[] pola_1 = {1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1};
        int[] pola_2 = {1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1};
        int[] pola_3 = {1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1};
        int[] pola_4 = {1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1};
        int[] pola_5 = {1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1};
        int[] pola_6 = {1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1};

        int[][] input = {pola_1, pola_2, pola_3, pola_4, pola_5, pola_6};
        char[] label = {'A', 'B', 'C', 'A', 'B', 'C'};
        int[] target = {1, -1, -1, 1, -1, -1};
        double alpha = 1;
        double theta = 0.5;

        int size = pola_1.length;
        double[] weight = new double[size];//model 

        int epoch = 0;
        boolean training = true;

        //HEADER
        System.out.print("Character;");
        for (int i = 0; i < size; i++) {
            System.out.print("x" + i + ";");
        }
        System.out.print("target;");
        for (int i = 0; i < size; i++) {
            System.out.print("x" + i + "*w" + i + ";");
        }
        System.out.print("net;");
        System.out.print("y;");
        System.out.print("y != t;");
        for (int i = 0; i < size; i++) {
            System.out.print("deltaW" + i + ";");
        }
        for (int i = 0; i < size; i++) {
            System.out.print("w" + i + ";");
        }
        System.out.println();

        while (training) {
            System.out.println("EPOCH-" + (++epoch) + "------------------------------------------------------------");
            int sama = 0;
            for (int j = 0; j < input.length; j++) {
                int[] x = input[j];
                int t = target[j];

                System.out.print(label[j] + ";");
                for (int i = 0; i < size; i++) {
                    int xi = x[i];
                    System.out.print(xi + ";");
                }
                System.out.print(t + ";");

                //SUMMATION---------------------------
                double net = 0;
                for (int i = 0; i < size; i++) {
                    double xi = x[i];
                    double wi = weight[i];
                    double xiwi = xi * wi;
                    //net = net + xiwi;
                    net += xiwi;
                    System.out.print(xiwi + ";");
                }
                System.out.print(net + ";");

                //ACTIVATION--------------------------
                //theta = 0.5
                //y = output = f(net)
                int y = -2;
                if (net > theta) {
                    y = 1;
                } else if (net >= -theta) {
                    y = 0;
                } else {//else if(net<-theta)
                    y = -1;
                }
                System.out.print(y + ";");
                //jika y != t maka lakukan update bobot dengan menghitung delta bobot
                double[] deltaWeight = new double[size];
                if (y != t) {
                    for (int i = 0; i < size; i++) {
                        double xi = x[i];
                        double deltaW = alpha * xi * t;
                        deltaWeight[i] = deltaW;
                    }
                    System.out.print("1;");
                } else {
                    sama++;//increment t yang sudah sama dengan y
                    System.out.print("0;");
                }

                //UPDATE BOBOT----------------------------------
                for (int i = 0; i < size; i++) {
                    System.out.print(deltaWeight[i] + ";");
                    //weight[i] = weight[i] + deltaWeight[i];
                    weight[i] += deltaWeight[i];
                }

                //PRINT BOBOT
                for (int i = 0; i < size; i++) {
                    System.out.print(weight[i] + ";");
                }

                System.out.println();
            }

            //kondisi berhenti
            if (sama == target.length) {
                //training = false;
                break;
            }
        }//end of TRAINING

        System.out.println("");
        System.out.println("TESTING");
        while (true) {
            System.out.println("PILIHAN: \n[0] = exit, \n[1] = testing");
            int pilihan = new Scanner(System.in).nextInt();
            if (pilihan == 0) {
                System.out.println("TESTING SELESAI");
                break;
            } else if (pilihan == 1) {
                System.out.println("testing");
                System.out.println("MASUKAN vector input");
                int[] xTest = {1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1};
        
                //int[] xTest = new int[size];                
//                for (int i = 0; i < size; i++) {
//                    xTest[i] = new Scanner(System.in).nextInt();
//                }

                //hitung net
                double netTest = 0;
                for (int i = 0; i < size; i++) {
                    double xi = xTest[i];
                    double wi = weight[i];
                    double xiwi = xi * wi;
                    netTest += xiwi;

                }

                int yTest = -2;
                if (netTest > theta) {
                    yTest = 1;
                } else if (netTest >= -theta) {
                    yTest = 0;
                } else {//else if(net<-theta)
                    yTest = -1;
                }
                
                System.out.println("OUTPUT HASIL RECOGNIZE");
                System.out.println("y = "+yTest);
                System.out.println("---------------------------------------");

            } else {
                System.out.println("PILIHAN tidak ada di menu Pilihan. pilih [0] atau [1]");
            }
        }

    }
}

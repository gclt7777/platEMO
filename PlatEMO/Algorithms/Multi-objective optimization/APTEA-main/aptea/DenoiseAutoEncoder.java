package jmetal.metaheuristics.aptea;

import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.metaheuristics.aptea.BaseAE.*;
import jmetal.util.JMException;
import jmetal.util.wrapper.XReal;

public class DenoiseAutoEncoder {
	
	int features;
	int feaAfterEncode;
	int layerNum;
	double learningRate;
	BPEncode baseAE;
	
	
	public DenoiseAutoEncoder(int numVars, int reD, int layerNum, double rate) {
		features = numVars;
		feaAfterEncode = reD;
		this.layerNum = layerNum;
		learningRate = rate;
		baseAE = new BPEncode(features,feaAfterEncode,layerNum,learningRate);
	}
	
	public void getTrainingModel(SolutionSet trainingSet, int epochs) throws JMException {
		int size = trainingSet.size();
		double[][] inputs = new double[size][features]; 
		for(int i=0;i<size;i++) {
			XReal sol = new XReal(trainingSet.get(i));
			for(int j=0;j<features;j++) {
				double low = sol.getLowerBound(j);
				double up = sol.getUpperBound(j);
				inputs[i][j] = (sol.getValue(j)-low)/(up-low);
			}
		}
		baseAE.trainModel_new(inputs, epochs);
	}
	
	public void getTrainingModel(SolutionSet trainingSet, double p, int epochs) throws JMException {
		int size = trainingSet.size();
		double[][] inputs = new double[size][features]; 
		for(int i=0;i<size;i++) {
			XReal sol = new XReal(trainingSet.get(i));
			for(int j=0;j<features;j++) {
				double low = sol.getLowerBound(j);
				double up = sol.getUpperBound(j);
				inputs[i][j] = (sol.getValue(j)-low)/(up-low);
			}
		}
		baseAE.trainModel(inputs, p, epochs);
	}
	
	public void getTrainingModel(SolutionSet trainingSet, int epochs, int[] group) throws JMException {
		int size = trainingSet.size();
		double[][] inputs = new double[size][features]; 
		for(int i=0;i<size;i++) {
			XReal sol = new XReal(trainingSet.get(i));
			for(int j=0;j<features;j++) {
				double low = sol.getLowerBound(group[j]);
				double up = sol.getUpperBound(group[j]);
				inputs[i][j] = (sol.getValue(group[j])-low)/(up-low);
			}
		}
		baseAE.trainModel_new(inputs, epochs);
	}
	
	public void getTrainingModel(SolutionSet trainingSet, double p, int epochs, int[] group) throws JMException {
		int size = trainingSet.size();
		double[][] inputs = new double[size][features]; 
		for(int i=0;i<size;i++) {
			XReal sol = new XReal(trainingSet.get(i));
			for(int j=0;j<features;j++) {
				double low = sol.getLowerBound(group[j]);
				double up = sol.getUpperBound(group[j]);
				inputs[i][j] = (sol.getValue(group[j])-low)/(up-low);
			}
		}
		baseAE.trainModel(inputs, p, epochs);
	}
	
	public double[] encode(Solution sol, int dim) throws JMException {
		double[] encodedSolution = new double[feaAfterEncode];
		XReal xsol = new XReal(sol);
		if(dim != features) {
			System.out.println("The variable-related dimensions of the input solution do not match the model at encode a single solution!");
			System.out.println("features = " + features + ", and the dimensions of input is: " + dim);
			System.exit(0);
		}else {
			double[] input = new double[features];
			for(int i=0;i<features;i++) { 
				double low = xsol.getLowerBound(i);
				double up = xsol.getUpperBound(i);
				input[i] = (xsol.getValue(i)-low)/(up-low); 
			}
			baseAE.computeEncodeData(input, encodedSolution);
		}
		return encodedSolution;
	}
	
	public double[][] encode(SolutionSet solSet,int dim) throws JMException{
		int size = solSet.size();
		double[][] encodedSet = new double[size][feaAfterEncode];
		for(int p=0;p<size;p++) {
			XReal xsol = new XReal(solSet.get(p));
			if(dim != features) {
				System.out.println("The variable-related dimensions of the input solution do not match the model at encode a solution set!");
				System.out.println("features = " + features + ", and the dimensions of input is: " + dim);
				System.exit(0);
			}else {
				double[] input = new double[features];
				for(int i=0;i<features;i++) { 
					double low = xsol.getLowerBound(i);
					double up = xsol.getUpperBound(i);
					input[i] = (xsol.getValue(i)-low)/(up-low); 
				}
				baseAE.computeEncodeData(input, encodedSet[p]);
			}
		}
		return encodedSet;
	}
	
	public double[] encode(double[] input) {
		double[] encodedSolution = new double[feaAfterEncode];
		if(input.length != features) {
			System.out.println("The variable-related dimensions of the input solution do not match the model at encode a array of input!");
			System.exit(0);
		}else {
			baseAE.computeEncodeData(input, encodedSolution);
		}
		return encodedSolution;
	}
	
	public double[] decode(double[] encodeData) {
		double[] decodedSolution = new double[features];
		if(encodeData.length != feaAfterEncode) {
			System.out.println("The dimensions of the input encoded data do not match the model at decode a arry of input!");
			System.exit(0);
		}else {
			baseAE.computeDecodeData(encodeData, decodedSolution);
		}
		return decodedSolution;
	}

}

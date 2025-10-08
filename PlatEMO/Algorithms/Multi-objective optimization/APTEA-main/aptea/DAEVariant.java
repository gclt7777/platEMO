package jmetal.metaheuristics.aptea;

import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.metaheuristics.aptea.BaseAE.*;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.util.wrapper.XReal;

public class DAEVariant {
	
	int features;
	int feaAfterEncode;
	int layerNum;
	double learningRate;
	BPAutoEncode baseAE;
	
	
	public DAEVariant (int numVars, int reD, int layerNum, double rate) {
		features = numVars;
		feaAfterEncode = reD;
		this.layerNum = layerNum;
		learningRate = rate;
		baseAE = new BPAutoEncode(features,feaAfterEncode,layerNum,learningRate);
	}
	
	public void getTrainingModel(SolutionSet trainingSet, SolutionSet targetSet, int epochs) throws JMException {
		int size = trainingSet.size();
		int tSize = targetSet.size();
		double[][] inputs = new double[size][features];
		double[][] targets = new double[size][features];
		for(int i=0;i<size;i++) {
			XReal sol = new XReal(trainingSet.get(i));
			for(int j=0;j<features;j++) {
				inputs[i][j] = sol.getValue(j);
			}
		}
		for(int s=0;s<epochs;s++) {
			for(int i=0;i<size;i++) {
				XReal tar;
				tar = new XReal(targetSet.get(PseudoRandom.randInt(0, tSize-1)));
				for(int j=0;j<features;j++) {
					targets[i][j] = tar.getValue(j);
				}
			}
			baseAE.trainModel(inputs, targets, 1);
		}
	}
	
	public void getTrainingModel(SolutionSet trainingSet, SolutionSet targetSet, double p, int epochs) throws JMException {
		int size = trainingSet.size();
		int tSize = targetSet.size();
		double[][] inputs = new double[size][features];
		double[][] targets = new double[size][features];
		for(int i=0;i<size;i++) {
			XReal sol = new XReal(trainingSet.get(i));
			for(int j=0;j<features;j++) {
				inputs[i][j] = sol.getValue(j);
			}
		}
		for(int s=0;s<epochs;s++) {
			for(int i=0;i<size;i++) {
				XReal tar;
				tar = new XReal(targetSet.get(PseudoRandom.randInt(0, tSize-1)));
				for(int j=0;j<features;j++) {
					targets[i][j] = tar.getValue(j);
				}
			}
			baseAE.trainModel(inputs, targets, p, 1);
		}
	}
	
	public void getTrainingModel(SolutionSet trainingSet, SolutionSet targetSet, int epochs, int[] group) throws JMException {
		int size = trainingSet.size();
		int tSize = targetSet.size();
		double[][] inputs = new double[size][features];
		double[][] targets = new double[size][features];
		for(int i=0;i<size;i++) {
			XReal sol = new XReal(trainingSet.get(i));
			XReal tar;
			if(tSize < size) {
				tar = new XReal(targetSet.get(PseudoRandom.randInt(0, tSize-1)));
			}else {
				tar = new XReal(targetSet.get(i));
			}
			for(int j=0;j<features;j++) {
				inputs[i][j] = sol.getValue(group[j]);
				targets[i][j] = tar.getValue(group[j]);
			}
		}
		baseAE.trainModel(inputs, targets, epochs);
	}
	
	public void getTrainingModel(SolutionSet trainingSet, SolutionSet targetSet, double p, int epochs, int[] group) throws JMException {
		int size = trainingSet.size();
		int tSize = targetSet.size();
		double[][] inputs = new double[size][features];
		double[][] targets = new double[size][features];
		for(int i=0;i<size;i++) {
			XReal sol = new XReal(trainingSet.get(i));
			XReal tar;
			if(tSize < size) {
				tar = new XReal(targetSet.get(PseudoRandom.randInt(0, tSize-1)));
			}else {
				tar = new XReal(targetSet.get(i));
			}
			for(int j=0;j<features;j++) {
				inputs[i][j] = sol.getValue(group[j]);
				targets[i][j] = tar.getValue(group[j]);
			}
		}
		baseAE.trainModel(inputs,targets, p, epochs);
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
				input[i] = xsol.getValue(i); 
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
					input[i] = xsol.getValue(i); 
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

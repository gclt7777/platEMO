package jmetal.metaheuristics.aptea;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.qualityIndicator.GenerationalDistance;
import jmetal.util.Distance;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.util.comparators.CrowdingComparator;
import jmetal.util.ranking.NondominatedRanking;
import jmetal.util.ranking.Ranking;
import jmetal.util.wrapper.XReal;

public class APTEAP extends Algorithm{
	int populationSize;
	int maxEvaluations;
	int evaluations;
	private SolutionSet previousPopulation_;
	SolutionSet population;
	SolutionSet offspringPopulation;
	SolutionSet union;
	
	Operator mutationOperator;
	Operator crossoverOperator;
	Operator selectionOperator;
	
	Distance distance = new Distance();
	
	int numVars;
	int numObjs;
	int[][] group;
	int numGroups;
	
	DAEVariant DAE;
	DenoiseAutoEncoder[] subDAE;
	double[] upBounds;
	double[] lowBounds;
	
	double[] zIdeal;
	double[] zNadir;
	
	GenerationalDistance gd_;
	
	/**
	 * Constructor
	 * 
	 * @param problem
	 *            Problem to solve
	 */
	public APTEAP(Problem problem) {
		super(problem);
	} // APTEA
	
	public void savePreviousPopulation() {
		previousPopulation_.clear();
		for(int i=0;i<populationSize;i++) {
			previousPopulation_.add(population.get(i));
		}
	}
	
	public void initialization() throws JMException, ClassNotFoundException {
		// Read the parameters
		populationSize = ((Integer) getInputParameter("populationSize")).intValue();
		maxEvaluations = ((Integer) getInputParameter("maxEvaluations")).intValue();
		
		mutationOperator = operators_.get("mutation");
		crossoverOperator = operators_.get("crossover");
		selectionOperator = operators_.get("selection");
		// Initialize the variables
		population = new SolutionSet(populationSize);
		previousPopulation_ = new SolutionSet(populationSize);
		offspringPopulation = new SolutionSet(populationSize);
		union = new SolutionSet(2*populationSize);;
		Solution newSolution;
		for (int j = 0; j < populationSize; j++) {
			newSolution = new Solution(problem_);
			problem_.evaluate(newSolution);
			problem_.evaluateConstraints(newSolution);
			population.add(newSolution);
		} // for		
		
		evaluations = 0;
		// Read the operators
		mutationOperator = operators_.get("mutation");
		crossoverOperator = operators_.get("crossover");
		selectionOperator = operators_.get("selection");
		
		numVars = problem_.getNumberOfVariables();
		numObjs = problem_.getNumberOfObjectives();
		
		upBounds = new double[numVars];
		lowBounds = new double[numVars];
		
	
		zIdeal = new double[numObjs];
		zNadir = new double[numObjs];
		//Construct the Auto-Encoders based on neural networks
		DAE = new DAEVariant(numVars, 1, 3, 0.1);
		SolutionSet[] st = getTrainingSet();
		DAE.getTrainingModel(st[0],st[1], 1);
		double[][][] weight = DAE.baseAE.getWeight();
		double[] W = new double[numVars];
		for(int var=0;var<numVars;var++) {
			W[var] = weight[0][var][0];
			upBounds[var] = problem_.getUpperLimit(var);
			lowBounds[var] = problem_.getLowerLimit(var);
		}
		//Variable grouping based on the weights
		numGroups = 2;
		group = new int[numGroups][];
		int mSize = numVars/numGroups;//mean
		int rSize = numVars%numGroups;//remainder
		for(int g=0;g<rSize;g++){
			group[g] = new int[mSize+1];
		}
		for(int g=rSize;g<numGroups;g++){
			group[g] = new int[mSize];
		}
		linearGrouping(W);
		//Construct the sub-Auto-Encoders based on neural networks
		subDAE = new DenoiseAutoEncoder[numGroups];
		for(int g=0;g<numGroups;g++) {
			subDAE[g] = new DenoiseAutoEncoder(group[g].length, 1, 3, 0.1);
			subDAE[g].getTrainingModel(st[1], 10, group[g]);
		}	
		
		gd_ = new GenerationalDistance();
		savePreviousPopulation();
	}
	
	public void linearGrouping(double[] W){
		
		int[] idx = new int[numVars];
		for(int i=0;i<numVars;i++) {
			idx[i] = i;
		}
		Utils.QuickSort(W, idx, 0, numVars-1);
		
		int gSize_ = numVars/numGroups;
		group = new int[numGroups][];
		for(int g=0;g<numGroups-1;g++){
			group[g] = new int[gSize_];
		}
		int lSize_ = numVars-(numGroups-1)*gSize_;//the variable size of the last group
		group[numGroups-1] = new int[lSize_];
		
		int t = 0;
		for(int g=0;g<numGroups-1;g++){
			for(int m=0;m<gSize_;m++){
				group[g][m] = idx[t];
				t++;
			}
		}
		//assign variable to the last group
		for(int m=0;m<lSize_;m++){
			group[numGroups-1][m] = idx[t];
			t++;
		}
	}

	public SolutionSet execute() throws JMException, ClassNotFoundException {
		initialization();
		while(evaluations <= maxEvaluations){
			if(evaluations % (populationSize*10) == 0) {
				SolutionSet[] st = getTrainingSet();
				DAE.getTrainingModel(st[0],st[1], 10);
				if(gd_.generationalDistance(previousPopulation_, population, numObjs) <= 0.01*numObjs && numGroups < numVars/5) {
					numGroups++;
					savePreviousPopulation();
					double[][][] weight = DAE.baseAE.getWeight();
					double[] W = new double[numVars];
					for(int var=0;var<numVars;var++) {
						W[var] = weight[0][var][0];
					}
					linearGrouping(W);
					if(evaluations % (populationSize*100) == 0) {
						System.out.println("The current group size is " + numGroups);
					}
				}
				//incrementally training the sub-Auto-Encoders based on neural networks for each task
				subDAE = new DenoiseAutoEncoder[numGroups];
				for(int g=0;g<numGroups;g++) {
					subDAE[g] = new DenoiseAutoEncoder(group[g].length, 1, 3, 0.1);
					subDAE[g].getTrainingModel(st[1], 10, group[g]);
				}	
			}
			reproduction();
			environmentalSelection();	
		}
		NondominatedRanking final_ranking = new NondominatedRanking(population);
		return final_ranking.getSubfront(0);
	}
	
	public void reproduction() throws JMException, ClassNotFoundException {
		Solution[] parents = new Solution[2];
		for (int i = 0; i < (populationSize/2); i++) {
			double rd = PseudoRandom.randDouble(0.0,1.0);
			// obtain parents
			parents[0] = (Solution) selectionOperator.execute(population);
			parents[1] = (Solution) selectionOperator.execute(population);
			if(rd < 0.25) {//search in the original n-dimensional search space
				Solution[] offSpring = (Solution[]) crossoverOperator.execute(parents);
				mutationOperator.execute(offSpring[0]);
				mutationOperator.execute(offSpring[1]);
				problem_.evaluate(offSpring[0]);
				problem_.evaluateConstraints(offSpring[0]);
				problem_.evaluate(offSpring[1]);
				problem_.evaluateConstraints(offSpring[1]);
				offSpring[0].setLearningType(1);
				offSpring[1].setLearningType(1);
				offspringPopulation.add(offSpring[0]);
				offspringPopulation.add(offSpring[1]);
			}else if(rd < 0.75) {//search in the K-dimensional representation space
				double rnd = PseudoRandom.randDouble(0.0,1.0);
				double[][] encoder = new double[2][];
				//Encode via the pertain sub-DAE models
				encoder = encodeWithSubDAE(parents);
				//Searching in the transfered subspace
				double[][] newEncoder = doCrossover(0.9, encoder);
				doMutation(newEncoder);
				//Decode via the pertain sub-DAE models
				Solution[] offSpring = decodeWithSubDAE(newEncoder);
				//Evaluation
				problem_.evaluate(offSpring[0]);
				problem_.evaluateConstraints(offSpring[0]);
				problem_.evaluate(offSpring[1]);
				problem_.evaluateConstraints(offSpring[1]);
				offSpring[0].setLearningType(2);
				offSpring[1].setLearningType(2);
				offspringPopulation.add(offSpring[0]);
				offspringPopulation.add(offSpring[1]);	
			}else {//search in the 1-dimensional representation space
				double rand = PseudoRandom.randDouble(0.0,1.0);
				double[][] encoder = new double[2][];
				//Encode via the pertain sub-DAE models
				encoder = encodeWithDAE(parents);
				//Decode via the pertain sub-DAE models
				Solution[] guiders = decodeWithDAE(encoder);
				Solution[] offSpring = new Solution[2];
				for(int p=0;p<2;p++) {
					offSpring[p] = new Solution(parents[p]);
					int rd1 = PseudoRandom.randInt(0, populationSize - 1);
					int rd2 = PseudoRandom.randInt(0, populationSize - 1);
					while(rd1 == rd2) {
						rd2 = PseudoRandom.randInt(0, populationSize - 1);
					}
					XReal xParent1 = new XReal(population.get(rd1));
					XReal xParent2 = new XReal(population.get(rd2));
					XReal xChild = new XReal(offSpring[p]);
					XReal xGuider = new XReal(guiders[p]);
					for (int j = 0; j < numVars; j++) {
						double value;
						if(PseudoRandom.randDouble() < 0.5) {
							value= xChild.getValue(j) + 0.5*(xParent1.getValue(j) - xParent2.getValue(j))
	                                + 0.5*(-xGuider.getValue(j) + xChild.getValue(j));
						}else {
							value= xChild.getValue(j) + 0.5*(xParent1.getValue(j) - xParent2.getValue(j));
						}
						if (value < lowBounds[j])
	                        value = lowBounds[j];
	                    if (value > upBounds[j])
	                        value = upBounds[j];
	                    xChild.setValue(j, value);
					}
				}
				//Evaluation
				problem_.evaluate(offSpring[0]);
				problem_.evaluateConstraints(offSpring[0]);
				problem_.evaluate(offSpring[1]);
				problem_.evaluateConstraints(offSpring[1]);
				offSpring[0].setLearningType(3);
				offSpring[1].setLearningType(3);
				offspringPopulation.add(offSpring[0]);
				offspringPopulation.add(offSpring[1]);
			}
			evaluations += 2;
		}
	}
	
	public double[][] encodeWithSubDAE(Solution[] parents) throws JMException {
		double[][][] encodedParents = new double[parents.length][][];
		for(int i=0;i<parents.length;i++) {
			encodedParents[i] = new double[numGroups][];
			XReal xsol = new XReal(parents[i]);
			for(int g=0;g<numGroups;g++) {
				int len = group[g].length;
				double[] input = new double[len];
				for(int j=0;j<len;j++) {
					int var = group[g][j];
					input[j] = (xsol.getValue(var)-lowBounds[var])/(upBounds[var]-lowBounds[var]);
				}
				encodedParents[i][g] = new double[1];
				encodedParents[i][g] = subDAE[g].encode(input);
			}
		}
		double[][] encoder = new double[2][numGroups];
		for(int p=0;p<2;p++) {
			for(int g=0;g<numGroups;g++) {
				encoder[p][g] = encodedParents[p][g][0];
			}
		}
		return encoder;
	}
	
	public double[] encodeWithSubDAE(Solution parent) throws JMException {
		double[][] encodedParents = new double[numGroups][];
		XReal xsol = new XReal(parent);
		for(int g=0;g<numGroups;g++) {
			int len = group[g].length;
			double[] input = new double[len];
			for(int j=0;j<len;j++) {
				int var = group[g][j];
				input[j] = (xsol.getValue(var)-lowBounds[var])/(upBounds[var]-lowBounds[var]);
			}
			encodedParents[g] = new double[1];
			encodedParents[g] = subDAE[g].encode(input);
		}
		double[] encoder = new double[numGroups];
		for(int g=0;g<numGroups;g++) {
			encoder[g] = encodedParents[g][0];
		}
		return encoder;
	}
	
	public Solution[] decodeWithSubDAE(double[][] encoder) throws JMException, ClassNotFoundException {
		Solution[] offspring = new Solution[2];
		XReal[] child = new XReal[2];
		for(int i=0;i<2;i++) {
			offspring[i] = new Solution(problem_);
			child[i] = new XReal(offspring[i]);
			for(int g=0;g<numGroups;g++) {
				double[] newEncode = new double[1];
				newEncode[0] = encoder[i][g];
				double[] newDecode = subDAE[g].decode(newEncode);
				double value;
				int len = group[g].length;
				for(int d=0;d<len;d++) {
					int var = group[g][d];
					value = newDecode[d]*(upBounds[var] - lowBounds[var]) + lowBounds[var];
					if(value < lowBounds[var]) {
						value = lowBounds[var];
					}
					if(value > upBounds[var]) {
						value = upBounds[var];
					}
					child[i].setValue(var, value);
				}
			}
		}
		return offspring;
	}
	
	public double[][] encodeWithDAE(SolutionSet solSet) throws JMException {
		int size = solSet.size();
		double[][] encoder = new double[size][];
		for(int i=0;i<size;i++) {
			encoder[i] = DAE.encode(solSet.get(i),numVars);
		}
		return encoder;
	}
	
	public double[][] encodeWithDAE(Solution[] parents) throws JMException {
		int len = parents.length;
		double[][] encoder = new double[len][];
		for(int i=0;i<len;i++) {
			encoder[i] = DAE.encode(parents[i],numVars);
		}
		return encoder;
	}
	
	public double[] encodeWithDAE(Solution parents) throws JMException {
		double[] encoder = new double[1];
		encoder = DAE.encode(parents,numVars);
		return encoder;
	}
	
	public Solution[] decodeWithDAE(double[][] encoder) throws JMException, ClassNotFoundException {
		Solution[] offspring = new Solution[2];
		XReal[] child = new XReal[2];
		for(int i=0;i<2;i++) {
			offspring[i] = new Solution(problem_);
			child[i] = new XReal(offspring[i]);
			double[] newEncode = new double[1];
			newEncode[0] = encoder[i][0];
			double[] newDecode = DAE.decode(newEncode);
			double value;
			for(int var=0;var<numVars;var++) {
				value = newDecode[var]*(upBounds[var] - lowBounds[var]) + lowBounds[var];
				if(value < lowBounds[var]) {
					value = lowBounds[var];
				}
				if(value > upBounds[var]) {
					value = upBounds[var];
				}
				child[i].setValue(var, value);
			}
		}
		return offspring;
	}
	
	public double[][] doCrossover(double probability, double[][] encodedParents) throws JMException {

		double[][] offs = new double[2][];
		/**
		 * EPS defines the minimum difference allowed between real values
		 */
		double EPS = 1.0e-14;
		int i;
		double rand;
		double y1, y2, yL, yu;
		double c1, c2;
		double alpha, beta, betaq;
		double valueX1, valueX2;
		double[] x1 = encodedParents[0];
		double[] x2 = encodedParents[1];
		int dim = encodedParents[0].length;
		double distributionIndex_ = 20.0;
		offs[0] = new double[dim];
		offs[1] = new double[dim];

		if (PseudoRandom.randDouble() <= probability) {
			for (i = 0; i < dim; i++) {
				valueX1 = x1[i];
				valueX2 = x2[i];
				if (PseudoRandom.randDouble() <= 0.5) {
					if (java.lang.Math.abs(valueX1 - valueX2) > EPS) {

						if (valueX1 < valueX2) {
							y1 = valueX1;
							y2 = valueX2;
						} else {
							y1 = valueX2;
							y2 = valueX1;
						} // if

						yL = 0.0;
						yu = 1.0;
						rand = PseudoRandom.randDouble();
						beta = 1.0 + (2.0 * (y1 - yL) / (y2 - y1));					
						
						alpha = 2.0 - java.lang.Math.pow(beta, -(distributionIndex_ + 1.0));

						if (rand <= (1.0 / alpha)) {
							betaq = java.lang.Math.pow((rand * alpha), (1.0 / (distributionIndex_ + 1.0)));
						} else {
							betaq = java.lang.Math.pow((1.0 / (2.0 - rand * alpha)),
									(1.0 / (distributionIndex_ + 1.0)));
						} // if
						if(Double.isNaN(betaq))
							System.out.println(java.lang.Math.pow(0.1,1.0/2.0));
						c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));

						beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1));
						alpha = 2.0 - java.lang.Math.pow(beta, -(distributionIndex_ + 1.0));

						if (rand <= (1.0 / alpha)) {
							betaq = java.lang.Math.pow((rand * alpha), (1.0 / (distributionIndex_ + 1.0)));
						} else {
							betaq = java.lang.Math.pow((1.0 / (2.0 - rand * alpha)),
									(1.0 / (distributionIndex_ + 1.0)));
						} // if

						c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

						if(Double.isNaN(c2))
							System.out.println(c2);
						if (c1 < yL)
							c1 = yL;

						if (c2 < yL)
							c2 = yL;

						if (c1 > yu)
							c1 = yu;

						if (c2 > yu)
							c2 = yu;

						if (PseudoRandom.randDouble() <= 0.5) {
							offs[0][i] = c2;
							offs[1][i] = c1;
						} else {
							offs[0][i] = c1;
							offs[1][i] = c2;
						} // if
					} else {
						offs[0][i] = valueX1;
						offs[1][i] = valueX2;
					} // if
				} else {
					offs[0][i] = valueX2;
					offs[1][i] = valueX1;
				} // if
			} // if
		} // if

		return offs;
	} // doCrossover
	
	public void doMutation(double[][] newEncoder) throws JMException { 
		for(int i=0;i<newEncoder.length;i++) {
			double probability = 1.0/newEncoder[i].length;
			if(newEncoder[i].length == 1) {
				probability = 0.2;
			}
			double rnd, delta1, delta2, mut_pow, deltaq;
			double y, yl, yu, val, xy;
			double[] x = newEncoder[i];
			for (int var = 0; var < x.length; var++) {
				if (PseudoRandom.randDouble() <= probability) {
					y = x[var];
					yl = 0.0;
					yu = 1.0;
					delta1 = (y - yl) / (yu - yl);
					delta2 = (yu - y) / (yu - yl);
					rnd = PseudoRandom.randDouble();
					mut_pow = 1.0 / (20.0 + 1.0);
					if (rnd <= 0.5) {
						xy = 1.0 - delta1;
						val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (Math.pow(xy, (20.0 + 1.0)));
						deltaq = java.lang.Math.pow(val, mut_pow) - 1.0;
					} else {
						xy = 1.0 - delta2;
						val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (java.lang.Math.pow(xy, (20.0 + 1.0)));
						deltaq = 1.0 - (java.lang.Math.pow(val, mut_pow));
					}
					y = y + deltaq * (yu - yl);
					if (y < yl)
						y = yl;
					if (y > yu)
						y = yu;
					x[var] = y;
				}
			} // for
		}
	} // doMutation
	
	public void environmentalSelection() {
		// Create the solutionSet union of solutionSet and offSpring
		union = ((SolutionSet) population).union(offspringPopulation);
		population.clear();
		offspringPopulation.clear();
		// Ranking the union
		NondominatedRanking ranking = new NondominatedRanking(union);
		int remain = populationSize;
		int index = 0;
		SolutionSet front = null;
		// Obtain the next front
		front = ranking.getSubfront(index);
		while ((remain > 0) && (remain >= front.size())) {
			// Assign crowding distance to individuals
			distance.crowdingDistanceAssignment(front, problem_.getNumberOfObjectives());
			// Add the individuals of this front
			for (int k = 0; k < front.size(); k++) {
				population.add(front.get(k));
			} // for

			// Decrement remain
			remain = remain - front.size();

			// Obtain the next front
			index++;
			if (remain > 0) {
				front = ranking.getSubfront(index);
			} // if
		} // while
		// Remain is less than front(index).size, insert only the best one
		if (remain > 0) { // front contains individuals to insert
			distance.crowdingDistanceAssignment(front, problem_.getNumberOfObjectives());
			front.sort(new CrowdingComparator());
			for (int k = 0; k < remain; k++) {
				population.add(front.get(k));
			} // for
			remain = 0;
		} // if				
	}//environmetalSelection
	
	public SolutionSet getStSolutionSet(SolutionSet ss,int size) {
		Ranking ranking = new NondominatedRanking(ss);
		int remain = size;
		int index = 0;
		SolutionSet front = null;
		SolutionSet mgPopulation = new SolutionSet();
		front = ranking.getSubfront(index);
		while ((remain > 0) && (remain >= front.size())) {

			for (int k = 0; k < front.size(); k++) {
				mgPopulation.add(front.get(k));
			} // for
			// Decrement remain
			remain = remain - front.size();
			// Obtain the next front
			index++;
			if (remain > 0) {
				front = ranking.getSubfront(index);
			} // if
		}
		if (remain > 0) { // front contains individuals to insert
			for (int k = 0; k < front.size(); k++) {
				mgPopulation.add(front.get(k));
			}
		}
		return mgPopulation;
	}
	
	public void estimateIdealNadirPoint(SolutionSet solSet){
		for(int p=0;p<solSet.size();p++){
			for(int i=0;i<numObjs;i++){
				if(solSet.get(p).getObjective(i) < zIdeal[i])
					zIdeal[i] = solSet.get(p).getObjective(i);
				if(solSet.get(p).getObjective(i) > zNadir[i])
					zNadir[i] = solSet.get(p).getObjective(i);
			}
		}
	}
	
	public void normalization(SolutionSet solSet){
		double value;
		for(int p=0;p<solSet.size();p++){
			double sum = 0.0;
			
			for(int i=0;i<numObjs;i++){
				value = (solSet.get(p).getObjective(i)-zIdeal[i])/(zNadir[i]-zIdeal[i]);
				//value = (solSet.get(p).getObjective(i)-zIdeal[i]);
				solSet.get(p).setNormalizedObjective(i, value);
				sum += value;
			}
			solSet.get(p).setSumValue(sum);
			
			for(int i=0;i<numObjs;i++){
				solSet.get(p).setIthTranslatedObjective(i, solSet.get(p).getNormalizedObjective(i)/sum);
			}
		}
	}

	public SolutionSet[] getTrainingSet(){
		SolutionSet[] st = new SolutionSet[2];
		NondominatedRanking ranking = new NondominatedRanking(population);
		int size = populationSize/2;
		st[0] = new SolutionSet();
		st[1] = new SolutionSet();
		SolutionSet front = ranking.getSubfront(0);
		Solution[] elites;
		if(front.size() > size) {
			distance.crowdingDistanceAssignment(front, numObjs);
			front.sort(new CrowdingComparator());
			elites = new Solution[size];
			for(int i=0;i<size;i++) {
				elites[i] = front.get(i);
			}
			for(int i=size;i<front.size();i++) {
				st[0].add(front.get(i));
			}	
		}else {
			elites = new Solution[front.size()];
			for(int i=0;i<front.size();i++) {
				elites[i] = front.get(i);
			}
		}
		
		int remain = populationSize/2 - st[0].size();
		int index = ranking.getNumberOfSubfronts()-1;
		front = ranking.getSubfront(index);
		while(remain > 0 && (remain >= front.size())) {
			distance.crowdingDistanceAssignment(front, numObjs);
			front.sort(new CrowdingComparator());
			for (int k = front.size()-1; k >= 0; k--) {
				st[0].add(front.get(k));
			} // for
			remain = remain - front.size();
			index--;
			if (remain > 0) {
				front = ranking.getSubfront(index);
			} // if
		}
		
		if (remain > 0) {
			distance.crowdingDistanceAssignment(front, numObjs);
			front.sort(new CrowdingComparator());
			for (int k = front.size()-1; k >= front.size()-remain; k--) {
				st[0].add(front.get(k));
			} // for
			remain = 0;
		} // if
		
		for(int i=0;i<size;i++) {
			Solution sol = st[0].get(i);
			double minDis = distance(sol, elites[0]);
			int minIndex = 0;
			for(int j=1;j<elites.length;j++) {
				double dis = distance(sol, elites[j]);
				if(dis < minDis) {
					minDis = dis;
					minIndex = j;
				}
			}
			st[1].add(elites[minIndex]);
		}

		return st;
	}
	public double distance(Solution s1, Solution s2) {
		double dis = 0.0;
		for(int i=0;i<numObjs;i++) {
			dis += (s1.getObjective(i) - s2.getObjective(i)) * (s1.getObjective(i) - s2.getObjective(i));
		}
		dis = Math.sqrt(dis);
		return dis;
	}
}

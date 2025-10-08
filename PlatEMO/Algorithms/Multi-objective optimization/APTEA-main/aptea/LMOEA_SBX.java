package jmetal.metaheuristics.aptea;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.util.Distance;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.util.comparators.CrowdingComparator;
import jmetal.util.ranking.NondominatedRanking;
import jmetal.util.wrapper.XReal;

public class LMOEA_SBX extends Algorithm{
	int populationSize;
	int maxEvaluations;
	int evaluations;
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
	
	DenoiseAutoEncoder DAE;
	DenoiseAutoEncoder[] subDAE;
	double[] W;
	double[] upBounds;
	double[] lowBounds;
	
	double[] productivity;
	
	/**
	 * Constructor
	 * 
	 * @param problem
	 *            Problem to solve
	 */
	public LMOEA_SBX(Problem problem) {
		super(problem);
	} // LMOEA
	
	public void initialization() throws JMException, ClassNotFoundException {
		// Read the parameters
		populationSize = ((Integer) getInputParameter("populationSize")).intValue();
		maxEvaluations = ((Integer) getInputParameter("maxEvaluations")).intValue();
		
		mutationOperator = operators_.get("mutation");
		crossoverOperator = operators_.get("crossover");
		selectionOperator = operators_.get("selection");
		// Initialize the variables
		population = new SolutionSet(populationSize);
		offspringPopulation = new SolutionSet(populationSize);
		union = new SolutionSet(2*populationSize);;
		Solution newSolution;
		for (int j = 0; j < populationSize; j++) {
			newSolution = new Solution(problem_);
			problem_.evaluate(newSolution);
			problem_.evaluateConstraints(newSolution);
			population.add(newSolution);
		} // for		
		
		productivity = new double[2];
		productivity[0] = 0.3;
		productivity[1] = 0.7;
		
		evaluations = 0;
		// Read the operators
		mutationOperator = operators_.get("mutation");
		crossoverOperator = operators_.get("crossover");
		selectionOperator = operators_.get("selection");
		
		numVars = problem_.getNumberOfVariables();
		numObjs = problem_.getNumberOfObjectives();
		W = new double[populationSize];
		
		upBounds = new double[numVars];
		lowBounds = new double[numVars];

		//Construct the Auto-Encoders based on neural networks for each task
		DAE = new DenoiseAutoEncoder(numVars, 1, 3, 0.05);
		DAE.getTrainingModel(population, 100);
		double[][][] weight = DAE.baseAE.getWeight();
		W = new double[numVars];
		for(int var=0;var<numVars;var++) {
			W[var] = weight[0][var][0];
			upBounds[var] = problem_.getUpperLimit(var);
			lowBounds[var] = problem_.getLowerLimit(var);
		}
		//Variable grouping based on the weights
		numGroups = 10;
		group = new int[numGroups][];
		int mSize = numVars/numGroups;//mean
		int rSize = numVars%numGroups;//remainder
		for(int g=0;g<rSize;g++){
			group[g] = new int[mSize+1];
		}
		for(int g=rSize;g<numGroups;g++){
			group[g] = new int[mSize];
		}
		linearGrouping();
		//Construct the sub-Auto-Encoders based on neural networks for each task
		subDAE = new DenoiseAutoEncoder[numGroups];
		for(int g=0;g<numGroups;g++) {
			subDAE[g] = new DenoiseAutoEncoder(group[g].length, 1, 3, 0.05);
		}
		
	}
	
	public void linearGrouping(){
		int[] idx = new int[numVars];
		for(int i=0;i<numVars;i++) {
			idx[i] = i;
		}
		Utils.QuickSort(W, idx, 0, numVars-1);
		int iter = 0;
		for(int g=0;g<numGroups;g++){
			for(int m=0;m<group[g].length;m++){
				group[g][m] = idx[iter];
				iter++;
			}
		}
	}

	public SolutionSet execute() throws JMException, ClassNotFoundException {
		initialization();
		while(evaluations <= maxEvaluations){
			DAE.getTrainingModel(population, 10);
			/*double[][][] weight = DAE.baseAE.getWeight();
			for(int var=0;var<numVars;var++) {
				W[var] = weight[0][var][0];
			}
			linearGrouping();*/
			//incrementally training the sub-Auto-Encoders based on neural networks for each task
			for(int g=0;g<numGroups;g++) {
				subDAE[g].getTrainingModel(population, 10, group[g]);
			}
			
			reproduction();
			environmentalSelection();	
			
			int s1,s2,s3;
			s1=s2=s3=0;
			for(int i=0;i<populationSize;i++) {
				if(population.get(i).getLearningType() == 1) s1++;
				if(population.get(i).getLearningType() == 2) s2++;
				if(population.get(i).getLearningType() == 3) s3++;
				if(s1+s2+s3 == 0) {
					productivity[0] = 1.0/2.5;
					productivity[1] = 1.0/2.5;
				}else {
					productivity[0] = 0.5*(productivity[0] + (double)s1/(s1+s2+s3));
					productivity[1] = 0.5*(productivity[1] + (double)(s1+s2)/(s1+s2+s3));
				}
			}
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
			if(rd < productivity[0]) {//rd < productivity[t][0]
				//search in the original search space
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
			}else if(rd < productivity[1]) {//search in the K-dimensional subspace
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
			}else {//search in the 1-Dimensional space
				double rand = PseudoRandom.randDouble(0.0,1.0);
				double[][] encoder = new double[2][];
				//Encode via the pertain sub-DAE models
				encoder = encodeWithDAE(parents);
				//Searching in the transfered subspace
				double[][] newEncoder = doCrossover(0.9, encoder);
				doMutation(newEncoder);
				//Decode via the pertain sub-DAE models
				Solution[] offSpring = decodeWithDAE(newEncoder);
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
}

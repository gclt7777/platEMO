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

public class LMEA_SBX2 extends Algorithm{
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
	
	DAEVariant DAE;
	double[] upBounds;
	double[] lowBounds;
	
	double productivity;
	
	/**
	 * Constructor
	 * 
	 * @param problem
	 *            Problem to solve
	 */
	public LMEA_SBX2(Problem problem) {
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
		
		productivity = 0.3;
		
		evaluations = 0;
		// Read the operators
		mutationOperator = operators_.get("mutation");
		crossoverOperator = operators_.get("crossover");
		selectionOperator = operators_.get("selection");
		
		numVars = problem_.getNumberOfVariables();
		numObjs = problem_.getNumberOfObjectives();
		
		upBounds = new double[numVars];
		lowBounds = new double[numVars];

		//Construct the Auto-Encoders based on neural networks for each task
		DAE = new DAEVariant(numVars, 10, 3, 0.1);
		SolutionSet[] st = getTrainingSet();
		DAE.getTrainingModel(st[0],st[1], 10);
		for(int var=0;var<numVars;var++) {
			upBounds[var] = problem_.getUpperLimit(var);
			lowBounds[var] = problem_.getLowerLimit(var);
		}
	}
	
	public SolutionSet execute() throws JMException, ClassNotFoundException {
		initialization();
		while(evaluations <= maxEvaluations){
			if(evaluations % (populationSize*100) == 0) {
				SolutionSet[] st = getTrainingSet();
				DAE.getTrainingModel(st[0],st[1], 10);
			}
			
			reproduction();
			environmentalSelection();	
			
			int s1,s2;
			s1=s2=0;
			for(int i=0;i<populationSize;i++) {
				if(population.get(i).getLearningType() == 1) s1++;
				if(population.get(i).getLearningType() == 2) s2++;
				if(s1+s2 == 0) {
					productivity = 0.4;
				}else {
					productivity = 0.5*(productivity + (double)s1/(s1+s2));
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
			if(rd < productivity) {//rd < productivity[t][0]
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
			}else {//search in the K-Dimensional space
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
			//double[] newEncode = new double[10];
			//newEncode[0] = encoder[i][0];
			double[] newDecode = DAE.decode(encoder[i]);
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
	
	public SolutionSet[] getTrainingSet(){
		SolutionSet[] st = new SolutionSet[2];
		NondominatedRanking ranking = new NondominatedRanking(population);
		int size = populationSize/2;
		st[0] = new SolutionSet(size);
		st[1] = new SolutionSet(size);
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

package jmetal.metaheuristics.aptea;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.core.Variable;
import jmetal.qualityIndicator.GenerationalDistance;
import jmetal.util.Distance;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.util.comparators.CrowdingComparator;
import jmetal.util.ranking.NondominatedRanking;
import jmetal.util.ranking.Ranking;
import jmetal.util.vector.TwoLevelWeightVectorGenerator;
import jmetal.util.vector.VectorGenerator;
import jmetal.util.wrapper.XReal;

public class LAPTEAP extends Algorithm{
	
	private int populationSize;//population size
	private int numObj_; //number of objectives
	private int numVar_; //number of variables
	
	/**
	 * Stores the population
	 */
	private SolutionSet previousPopulation_;
	private SolutionSet population_;
	private SolutionSet offspringPopulation_;
	private SolutionSet unionPopulation_;	
	private int evaluations;
	private int maxEvaluations;	
	/**
	 * Operators
	 */
	private Operator crossoverOperator_;
	private Operator mutationOperator_;	
	private int numGroups; //number of groups
	private int[][] group_;
	double[] zideal_;
	double[] nadir_;
	double[] UB_;
	double[] LB_;
	
	DAEVariant DAE;
	
	GenerationalDistance gd_;
	
	Distance distance = new Distance();
	
	public LAPTEAP(Problem problem) {
		super(problem);
		numObj_ = problem.getNumberOfObjectives();
		numVar_ = problem.getNumberOfVariables();
	}

	public SolutionSet execute() throws JMException, ClassNotFoundException {
		initialization();
		while(evaluations <= maxEvaluations){
			if(evaluations % (populationSize*10) == 0) {
				SolutionSet[] st = getTrainingSet();
				DAE.getTrainingModel(st[0],st[1], 10);
				if(gd_.generationalDistance(previousPopulation_, population_, numObj_) <= 0.01*numObj_ && numGroups < numVar_/5) {
					numGroups++;
					double[][][] weight = DAE.baseAE.getWeight();
					double[] W = new double[numVar_];
					for(int var=0;var<numVar_;var++) {
						W[var] = weight[0][var][0];
					}
					linearGrouping(W);
					if(evaluations % (populationSize*100) == 0) {
						System.out.println("The current group size is " + numGroups);
					}
				} 
				savePreviousPopulation();
			}
			reproduction();
			environmentalSelection();	
			evaluations += populationSize;
		}
		NondominatedRanking final_ranking = new NondominatedRanking(population_);
		return final_ranking.getSubfront(0);
	}
	
	public void savePreviousPopulation() {
		previousPopulation_.clear();
		for(int i=0;i<populationSize;i++) {
			previousPopulation_.add(population_.get(i));
		}
	}
	
	public void initialization() throws JMException, ClassNotFoundException{
		evaluations = 0;
		populationSize = ((Integer) getInputParameter("populationSize")).intValue();
		maxEvaluations = ((Integer) getInputParameter("maxEvaluations")).intValue();
		population_ = new SolutionSet(populationSize);
		previousPopulation_ = new SolutionSet(populationSize);
		offspringPopulation_ = new SolutionSet(populationSize);
		unionPopulation_ = new SolutionSet(2*populationSize);
		//Read the operators
		mutationOperator_ = operators_.get("mutation");
		crossoverOperator_ = operators_.get("crossover");
		//Create the initial population
		Solution newSolution;
		for (int i = 0; i < populationSize; i++) {
			newSolution = new Solution(problem_);
			problem_.evaluate(newSolution);
			problem_.evaluateConstraints(newSolution);
			population_.add(newSolution);
		} // for
		zideal_ = new double[problem_.getNumberOfObjectives()];
		nadir_ = new double[problem_.getNumberOfObjectives()];
		UB_ = new double[numVar_];
		LB_ = new double[numVar_];
		
		//Construct the Auto-Encoders based on neural networks
		DAE = new DAEVariant(numVar_, 1, 3, 0.1);
		SolutionSet[] st = getTrainingSet();
		DAE.getTrainingModel(st[0],st[1], 1);
		double[][][] weight = DAE.baseAE.getWeight();
		double[] W = new double[numVar_];
		for(int var=0;var<numVar_;var++) {
			W[var] = weight[0][var][0];
			UB_[var] = problem_.getUpperLimit(var);
			LB_[var] = problem_.getLowerLimit(var);
		}		
		
		numGroups = 2;
		linearGrouping(W);
		gd_ = new GenerationalDistance();
		savePreviousPopulation();
	}
	
	public void reproduction() throws JMException{
		double learningRate;
		Solution[] parents = new Solution[3];
		for (int i = 0; i < population_.size(); i++) {
			// obtain parents
			matingSelection(parents,i);
			XReal[] xParents = new XReal[3];
			for(int p=0; p<3; p++) {
				xParents[p] = new XReal(parents[p]);
			}
			Solution child = new Solution(parents[0]);
			learningRate = PseudoRandom.randDouble(0.0, 1.0);
			//encode
			double[][] encodedParents = encode(xParents);
            //search in the compressed space
			double[] newEncode = searchInTransferedSpace(encodedParents,learningRate);
			//decode
			decode(newEncode,child);
			//evaluation of the child
			mutationOperator_.execute(child);
			problem_.evaluate(child);
			offspringPopulation_.add(child);
		}
	}//for reproduction
	
	public void matingSelection(Solution[] parents,int i) {
		parents[0] = population_.get(i);
		int rdInt1 = PseudoRandom.randInt(0, populationSize-1);
		while(rdInt1 == i) {
			rdInt1 = PseudoRandom.randInt(0, populationSize-1);
		}
		int rdInt2 = PseudoRandom.randInt(0, populationSize-1);
		while(rdInt2 == i || rdInt2 == rdInt1) {
			rdInt2 = PseudoRandom.randInt(0, populationSize-1);
		}
		parents[1] = population_.get(rdInt1);
		parents[2] = population_.get(rdInt2);
	}
	
	public double[][] encode(XReal[] parents) throws JMException {
		int len = parents.length;
		double[][] encodedParents = new double[len][numGroups];
		for(int l=0;l<len;l++) {
			double[] encodedSolution = new double[numVar_];
			encodedSolution = normalizedVariable(parents[l]);
			for(int g=0;g<numGroups;g++) {
				encodedParents[l][g] = 0;
				for(int var=0;var<group_[g].length;var++) {
					encodedParents[l][g] += encodedSolution[group_[g][var]];
				}
				encodedParents[l][g] = encodedParents[l][g]/group_[g].length;
			}
		}
		return encodedParents; 
	}
	
	public double[] searchInTransferedSpace(double[][] encodedParents, double learningRate) {
		double[] newEncode = new double[numGroups];
		for(int j=0;j<numGroups;j++) {
			newEncode[j] = encodedParents[0][j] + learningRate*(encodedParents[1][j] - encodedParents[2][j]);
			if(newEncode[j] < 0) { newEncode[j] = 0.00001; } 
			if(newEncode[j] > 1) { newEncode[j] = 0.99999; }
		}
		return newEncode;
	}
	
	public void decode(double[] newEncode, Solution child) throws JMException {
		XReal offspring = new XReal(child);
		for(int g=0;g<numGroups;g++) {
			double sum = 0;
			for(int var=0;var<group_[g].length;var++) {
				int cVar = group_[g][var];
				sum += (offspring.getValue(cVar)-LB_[cVar])/(UB_[cVar] - LB_[cVar]);
			}
			if(sum == 0) sum = 0.00001;
			for(int var=0;var<group_[g].length;var++) {
				int cVar = group_[g][var];
				double normalizedValue = (offspring.getValue(cVar)-LB_[cVar])/(UB_[cVar] - LB_[cVar]);
				double decodedValue = (normalizedValue/sum) * newEncode[g] * group_[g].length;
				decodedValue = decodedValue*(UB_[cVar] - LB_[cVar]) + LB_[cVar];
				if(decodedValue < LB_[cVar]) {
					decodedValue = LB_[cVar] + 0.00001;
				}
				if(decodedValue > UB_[cVar]) {
					decodedValue = UB_[cVar] - 0.00001;
				}
				offspring.setValue(cVar, decodedValue);
			}
		}
	}
	
	public double[] normalizedVariable(XReal solution) throws JMException {
		double[] x = new double[numVar_];
		for (int i = 0; i < numVar_; i++) 
			x[i] = solution.getValue(i);
		for (int i = 0; i < numVar_; i++) {
			x[i] = (x[i] - LB_[i])  / (UB_[i] - LB_[i]);
		}
		return x;
	}
	
	public void linearGrouping(double[] W){
		
		int[] idx = new int[numVar_];
		for(int i=0;i<numVar_;i++) {
			idx[i] = i;
		}
		Utils.QuickSort(W, idx, 0, numVar_-1);
		
		int gSize_ = numVar_/numGroups;
		group_ = new int[numGroups][];
		for(int g=0;g<numGroups-1;g++){
			group_[g] = new int[gSize_];
		}
		int lSize_ = numVar_-(numGroups-1)*gSize_;//the variable size of the last group
		group_[numGroups-1] = new int[lSize_];
		
		int t = 0;
		for(int g=0;g<numGroups-1;g++){
			for(int m=0;m<gSize_;m++){
				group_[g][m] = idx[t];
				t++;
			}
		}
		//assign variable to the last group
		for(int m=0;m<lSize_;m++){
			group_[numGroups-1][m] = idx[t];
			t++;
		}
	}
	
	public void randomGrouping(){
		int gSize_ = numVar_/numGroups;
		group_ = new int[numGroups][];
		for(int g=0;g<numGroups-1;g++){
			group_[g] = new int[gSize_];
		}
		int lSize_ = numVar_-(numGroups-1)*gSize_;//the variable size of the last group
		group_[numGroups-1] = new int[lSize_];
		int[] permutation = new int[numVar_];
		Utils.randomPermutation(permutation, numVar_);
		int t = 0;
		for(int g=0;g<numGroups-1;g++){
			for(int m=0;m<gSize_;m++){
				group_[g][m] = permutation[t];
				t++;
			}
		}
		//assign variable to the last group
		for(int m=0;m<lSize_;m++){
			group_[numGroups-1][m] = permutation[t];
			t++;
		}
	}
	
	public void linearGrouping(){
		int gSize_ = numVar_/numGroups;
		group_ = new int[numGroups][];
		for(int g=0;g<numGroups-1;g++){
			group_[g] = new int[gSize_];
		}
		int lSize_ = numVar_-(numGroups-1)*gSize_;//the variable size of the last group
		group_[numGroups-1] = new int[lSize_];
		
		int t = 0;
		for(int g=0;g<numGroups-1;g++){
			for(int m=0;m<gSize_;m++){
				group_[g][m] = t;
				t++;
			}
		}
		//assign variable to the last group
		for(int m=0;m<lSize_;m++){
			group_[numGroups-1][m] = t;
			t++;
		}
	}
	/*
	 * Estimate the Ideal Point 
	 */
	public void estimateIdealPoint(SolutionSet solutionSet){
		for(int i=0; i<numObj_;i++){
			zideal_[i] = 1.0e+30;
			for(int j=0; j<solutionSet.size();j++){
				if(solutionSet.get(j).getObjective(i) < zideal_[i]){
					zideal_[i] = solutionSet.get(j).getObjective(i);
				}//if
			}//for
		}//for
	}
	
	/*
	 * Estimate the Ideal Point 
	 */
	public void estimateNadirPoint(SolutionSet solutionSet){
		for(int i=0; i<numObj_;i++){
			nadir_[i] = -1.0e+30;
			for(int j=0; j<solutionSet.size();j++){
				if(solutionSet.get(j).getObjective(i) > nadir_[i]){
					nadir_[i] = solutionSet.get(j).getObjective(i);
				}//if
			}//for
		}//for
	}
	
	public void normalization(SolutionSet solSet){
		double value;
		for(int p=0;p<solSet.size();p++){
			double sum = 0.0;
			
			for(int i=0;i<numObj_;i++){
				value = (solSet.get(p).getObjective(i)-zideal_[i])/(nadir_[i]-zideal_[i]);
				//value = (solSet.get(p).getObjective(i)-zIdeal[i]);
				solSet.get(p).setNormalizedObjective(i, value);
				sum += value;
			}
			solSet.get(p).setSumValue(sum);
			
			for(int i=0;i<numObj_;i++){
				solSet.get(p).setIthTranslatedObjective(i, solSet.get(p).getNormalizedObjective(i)/sum);
			}
		}
	}
	
	
	public void environmentalSelection() {
		// Create the solutionSet union of solutionSet and offSpring
		unionPopulation_ = ((SolutionSet) population_).union(offspringPopulation_);
		population_.clear();
		offspringPopulation_.clear();
		// Ranking the union
		NondominatedRanking ranking = new NondominatedRanking(unionPopulation_);
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
				population_.add(front.get(k));
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
				population_.add(front.get(k));
			} // for
			remain = 0;
		} // if				
	}//environmetalSelection
	
	public SolutionSet[] getTrainingSet(){
		SolutionSet[] st = new SolutionSet[2];
		NondominatedRanking ranking = new NondominatedRanking(population_);
		int size = populationSize/2;
		st[0] = new SolutionSet();
		st[1] = new SolutionSet();
		SolutionSet front = ranking.getSubfront(0);
		Solution[] elites;
		if(front.size() > size) {
			distance.crowdingDistanceAssignment(front, numObj_);
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
			distance.crowdingDistanceAssignment(front, numObj_);
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
			distance.crowdingDistanceAssignment(front, numObj_);
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
	
	public double distance(Solution s1, Solution s2) {
		double dis = 0.0;
		for(int i=0;i<numObj_;i++) {
			dis += (s1.getObjective(i) - s2.getObjective(i)) * (s1.getObjective(i) - s2.getObjective(i));
		}
		dis = Math.sqrt(dis);
		return dis;
	}
}
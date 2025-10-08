package jmetal.metaheuristics.aptea;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.core.Variable;
import jmetal.metaheuristics.moead.Utils;
import jmetal.qualityIndicator.GenerationalDistance;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.Configuration;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.util.ranking.NondominatedRanking;
import jmetal.util.ranking.Ranking;
import jmetal.util.vector.TwoLevelWeightVectorGenerator;
import jmetal.util.vector.VectorGenerator;
import jmetal.util.wrapper.XReal;

public class MOEAC extends Algorithm{
	
	private int populationSize;//population size
	private int numObj_; //number of objectives
	private int numVar_; //number of variables
	
	/**
	 * Stores the population
	 */
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
	private Operator selectionOperator_;
	double[] zideal_;
	double[] nadir_;
	double[] UB_;
	double[] LB_;
	
	double[] IGD;
	QualityIndicator indicator; // Object to get quality indicators
	
	
	public MOEAC(Problem problem) {
		super(problem);
		numObj_ = problem.getNumberOfObjectives();
		numVar_ = problem.getNumberOfVariables();
	}
	
	public MOEAC(Problem problem, QualityIndicator indicator) {
		super(problem);
		numObj_ = problem.getNumberOfObjectives();
		numVar_ = problem.getNumberOfVariables();
		this.indicator = indicator;
	}

	public SolutionSet execute() throws JMException, ClassNotFoundException {
		initialization();
		int G = 0;
		while(evaluations <= maxEvaluations){
			
			reproduction();
			environmentalSelection();
			//if(evaluations % 1200 == 0) {
				//IGD[G] = indicator.getIGD1(population_);
				//G++;
			//}
			evaluations += populationSize;
			
		}
		//printGD("MOEAC_"+numObj_+"Obj_"+ problem_.getName() + "_" + numVar_ + "D_IGD.txt",IGD);
		NondominatedRanking final_ranking = new NondominatedRanking(population_);
		return final_ranking.getSubfront(0);
	}
	
	public void initialization() throws JMException, ClassNotFoundException{
		evaluations = 0;
		populationSize = ((Integer) getInputParameter("populationSize")).intValue();
		maxEvaluations = ((Integer) getInputParameter("maxEvaluations")).intValue();
		population_ = new SolutionSet(populationSize);
		offspringPopulation_ = new SolutionSet(populationSize);
		unionPopulation_ = new SolutionSet(2*populationSize);
		//Read the operators
		mutationOperator_ = operators_.get("mutation");
		crossoverOperator_ = operators_.get("crossover");
		selectionOperator_ = operators_.get("selection");
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
		for (int i = 0; i < numVar_; i++) {
			UB_[i] = problem_.getUpperLimit(i);
			LB_[i] = problem_.getLowerLimit(i);
		} // for
		
		IGD = new double[101];
	}
	
	public void reproduction() throws JMException{
		Solution[] parents = new Solution[2];
		for (int i = 0; i < population_.size(); i++) {
			parents = (Solution[]) selectionOperator_.execute(population_);
			Solution[] offSpring = (Solution[]) crossoverOperator_
					.execute(parents);
			mutationOperator_.execute(offSpring[0]);
			problem_.evaluate(offSpring[0]);
			problem_.evaluateConstraints(offSpring[0]);
			mutationOperator_.execute(offSpring[0]);
			problem_.evaluate(offSpring[0]);
			offspringPopulation_.add(offSpring[0]);
		}
	}//for reproduction
	
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
	
	public void bestSolutionSelection(List<SolutionSet> list){
    	
    	for(int k=0; k<problem_.getNumberOfObjectives();k++){
    		double minClustering2Axis = Math.acos(Math.abs(list.get(0).getCentroid().getNormalizedObjective(k)
    						/list.get(0).getCentroid().getDistanceToIdealPoint()));
      		int minClustering2AxisID = 0;
      		for(int i=1;i<list.size();i++){
      			SolutionSet sols = list.get(i);
      			if(sols.size() == 0){
      				System.out.println("ElitSolutionSelection_Diversity_SolsSize = "+sols.size());
      				System.exit(0);
      			}
      			
      			double angle1 = Math.acos(Math.abs(sols.getCentroid().getNormalizedObjective(k)/sols.getCentroid().getDistanceToIdealPoint()));
      			//System.out.println(angle1);
      			if(angle1 < minClustering2Axis){
      				minClustering2Axis = angle1;
      				minClustering2AxisID = i;
      			}//if
      		}//for
      		double minSolution2Axis = Math.acos(list.get(minClustering2AxisID).get(0).getNormalizedObjective(k)
      				/list.get(minClustering2AxisID).get(0).getDistanceToIdealPoint());;
      		int minSolution2AxisID = 0;
      		for(int j=1;j<list.get(minClustering2AxisID).size();j++){
      			Solution sol = list.get(minClustering2AxisID).get(j);
      			double ang = Math.acos(list.get(minClustering2AxisID).get(j).getNormalizedObjective(k)
      					/list.get(minClustering2AxisID).get(j).getDistanceToIdealPoint());
      			if(ang < minSolution2Axis){
      				minSolution2Axis = ang;
      				minSolution2AxisID = j;
      			}
      		}//for
      		double rnd = PseudoRandom.randDouble();
      		if(rnd < 0.9){
      			population_.add(list.get(minClustering2AxisID).get(minSolution2AxisID));
              	list.remove(minClustering2AxisID);
      		}
    	}
    	
    	
	   Iterator<SolutionSet> it = list.iterator();
		while(it.hasNext()){
			SolutionSet sols = it.next();
			if(sols.size() == 0){
				System.out.println("In best solution selection, SolsSize2 = "+sols.size());
				System.exit(0);
			}
			if(sols.size() == 0){
				System.out.println("size = 0!");
				System.exit(0);
			}
			double minFitness = sols.get(0).getSumValue();
			int minFitnessID = 0;
			for(int j=1;j<sols.size();j++){
				Solution sol2 = sols.get(j);
				double fitness = sol2.getSumValue();
				if(minFitness > fitness){
					minFitness = fitness;
					minFitnessID = j;
				}	
			}//for
			population_.add(sols.get(minFitnessID));
			it.remove();
		}//while
		if(list.size() != 0){
			System.out.println("In best solution selection, ListSize2 = "+list.size());
			System.exit(0);
		}
   }
	
	public void environmentalSelection(){
		// Create the solutionSet union of solutionSet and offSpring
		unionPopulation_ = ((SolutionSet) population_).union(offspringPopulation_);
		population_.clear();
		offspringPopulation_.clear();
		SolutionSet st = getStSolutionSet(unionPopulation_,(int)(1.1*populationSize));
		estimateIdealPoint(st);
		estimateNadirPoint(st);
		normalization(st);
		List<SolutionSet> list = new <SolutionSet>ArrayList();
		for(int i=0;i<st.size();i++){
			SolutionSet sols = new SolutionSet();
			sols.add(st.get(i));
		    list.add(sols);  
		}
		list = new HierarchicalClustering(list).clusteringAnalysis(populationSize);
		if(list.size() != populationSize){
			System.out.println("The number of clusters after hierarchical clustering: ListSize = "+list.size());
			System.exit(0);
		}
		bestSolutionSelection(list);
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
	
	public static void printGD(String path,double[] GD){
	    try {
	    	/* Open the file */
	    	FileOutputStream fos   = new FileOutputStream(path)     ;//java文件输出�?，创建文件�?
	    	OutputStreamWriter osw = new OutputStreamWriter(fos)    ;//OutputStreamWriter是字符�?通�?�字节�?的桥�? 
	    	BufferedWriter bw      = new BufferedWriter(osw)        ;//缓冲区               
	    	for (int i = 0; i < GD.length; i++) {  
	    		bw.write(GD[i]+" ");//写到缓冲区
	    		bw.newLine(); //�?�行       
	    	}
	      
	    	/* Close the file */
	    	bw.close();
	    }catch (IOException e) {
	    	Configuration.logger_.severe("Error acceding to the file");
	    	e.printStackTrace();
	    }       
	} // printGD
}
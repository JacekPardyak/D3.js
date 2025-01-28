// Java Program to Illustrate Usage of Weka API

// Importing required classes
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

// Main class
// BreastCancer
public class GFG {

	// Main driver method
	public static void main(String args[])
	{

		// Try block to check for exceptions
		try {

			// Create J48 classifier by
			// creating object of J48 class
			J48 j48Classifier = new J48();

			// Dataset path
			String breastCancerDataset
				= "/home/droid/Tools/weka-3-8-5/data/breast-cancer.arff";

			// Creating bufferedreader to read the dataset
			BufferedReader bufferedReader
				= new BufferedReader(
					new FileReader(breastCancerDataset));

			// Create dataset instances
			Instances datasetInstances
				= new Instances(bufferedReader);

			// Set Target Class
			datasetInstances.setClassIndex(
				datasetInstances.numAttributes() - 1);

			// Evaluating by creating object of Evaluation
			// class
			Evaluation evaluation
				= new Evaluation(datasetInstances);

			// Cross Validate Model with 10 folds
			evaluation.crossValidateModel(
				j48Classifier, datasetInstances, 10,
				new Random(1));

			System.out.println(evaluation.toSummaryString(
				"\nResults", false));
		}

		// Catch block to handle the exceptions
		catch (Exception e) {

			// Print message on the console
			System.out.println("Error Occurred!!!! \n"
							+ e.getMessage());
		}
	}
}

package edu.stanford.cs246.hw2;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.text.DecimalFormat;
import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
//import org.ejml.ops.CommonOps;
import org.ejml.simple.*;
//import org.ejml.simple.SimpleSVD;

/**
 * Algorithm:
 * 
 * Map: 
 * setup(){ 
 * 	Read the centroids 
 * } 
 * map(point) {
 * 
 * 	for each point: 
 * 		Find the nearest centroid 
 * 		emit nearest centroid :
 * 			<k,v> = <centroid id, point> 
 * 		emit cost:
 * 			 <k,v> = <-1, min_distance ^2 >
 * }
 * 
 * Reduce: 
 * setup(){ 
 * 	Read the centroids 
 * } 
 * 
 * reduce(key, points = [list of points]){ 
 * 	centroid_running_average = 0 
 * 	cost = 0 
 * 
 * 		for each point in points:
 *   		if key  == -1 :
 *   			add value to cost
 *   		else :
 * 				add point to centroid_running_average
 *  
 * 	emit cost : 
 * 		<k,v> = <C+centroid id, cost> 
 * 	emit new centroid : 
 * 		<k,v> = <K+centroid id, centroid_running_average/no of points> 
 * }
 * 
 * Main (centroid init file, input path, output dir){
 * 		for i in #jobs: 
 * 			if first iteration: 
 * 				pass centroid file path to mapper 
 * 				Run the job to get output in outputdir/i 
 * 			else: 
 * 				Merge all output in outputdir/(i-1) on HDFS to create a centroid(i) file 
 * 				pass centroid file path to mapper 
 * 				Run the job to get output in outputdir/i
 * }
 * 
 * @author sa
 * 
 */
public class KSC {

	public static String CFILE = "cFile";
	public static String NUMOFCLUSTERS="numOfClusters";
	public static String NUMITERATIONS="numOfIterations";
	public static DecimalFormat df = new DecimalFormat("#.#####");
	public static int NUMPOINTS = 96;



	public static class CentroidMapper extends
			Mapper<Object, Text, IntWritable, Text> {

		//public ArrayList<double[]> INIT = new ArrayList<double[]>();
		public double[][]INIT;
		public Log log=LogFactory.getLog(CentroidMapper.class);
		public int numPoints = 96;
		public int numOfClusters;
		public int numIterations;
		
		public SimpleMatrix p = new SimpleMatrix(1,numPoints,true,new double[numPoints]);
		public SimpleMatrix c;
		
		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {

			Configuration conf = context.getConfiguration();
			
			/***Uncomment for amazon  ***/	
			//String uriStr = "s3n://energydata/centroid/"; 
			String uriStr = "./centroid";
			
			log.info("Mapper started fs-get centroid file");
			URI uri = URI.create(uriStr);
			FileSystem fs = FileSystem.get(uri, context.getConfiguration());		

			Path cFile=new Path(fs.getConf().get(CFILE));

			//System.out.println("Mapper centroid input:" + cFile.toString());

			//System.out.println(cFile.toString());
		
			DataInputStream d = new DataInputStream(fs.open(cFile));
			BufferedReader reader = new BufferedReader(new InputStreamReader(d));
			String line;
			numOfClusters = Integer.valueOf(context.getConfiguration().get(NUMOFCLUSTERS));
			numIterations = Integer.valueOf(context.getConfiguration().get(NUMITERATIONS));
			log.info("Mapper started reading centroid file");
			INIT = new double[numOfClusters][numPoints];
			
			int centroidLength=0;
			while ((line = reader.readLine()) != null) {
				if (!line.startsWith("C") && !line.startsWith("s")&& !line.startsWith("w")&&centroidLength !=numOfClusters) {
					double[] one_point = new double[numPoints];
					one_point=(parsePoint(line));	
					for (int i=0;i<numPoints;i++){
						INIT[centroidLength][i] = one_point[i];
					}
					//System.out.println(String.valueOf(INIT[centroidLength]));
					centroidLength++;		
				}
			}
			log.info("Mapper finished reading centroid file");
			c = new SimpleMatrix(INIT);		
		}

		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {

			if (!value.toString().startsWith("s")){ //check that its not a header
				// 1. Get point and parse it
				log.info("Mapper started parsing input");
				double[] point = parsePoint(value.toString());
				String info  = parseInfo(value.toString()); //SPID, Date;
				log.info("Mapper finished parsing input");
				double dailyTotal = 0;
				for (int i=0; i<NUMPOINTS;i++){
					dailyTotal+=point[i];
				}
				// 3. Find the closest centroid for the point (omit daily totals that are zero)
				log.info("Mapper started calculating distance");
				if (dailyTotal >0){
					// 2. Get distance of point from each centroid
					int closestCentroid = 0;
					double distance = Long.MAX_VALUE;

					if (numIterations >0) {   //For the first iteration, assign clusters randomly
						//for (double[] centroid : INIT) {
						for (int centroidIndex=0;centroidIndex<numOfClusters;centroidIndex++){
							for (int i=0;i<numPoints;i++){	
								p.set(i,point[i]);
							}						
							SimpleMatrix mat = c.extractVector(true, centroidIndex);
							double tmp = distance(c.extractVector(true,centroidIndex), p);
							if (tmp < distance) {
								closestCentroid = centroidIndex; //INIT.indexOf(centroid);
								distance = tmp;
							}
						}
					}else{
						
						closestCentroid = Math.round((numOfClusters-1)*(float)Math.random());
						//System.out.println("Iteration #:" + numIterations + " #of Clusters:" + String.valueOf(numOfClusters) + " Closest Centroid:" + String.valueOf(closestCentroid));
					}
					log.info("Mapper finished calculating distance");
					
					// 4. Emit cluster id and point
					log.info("Mapper started writing point");
					context.write(new IntWritable(closestCentroid),
							new Text(longArrayToString(point)));
					//System.out.println(closestCentroid);
					log.info("Mapper finished writing point-start spid");
					//Write key=Centroid, Value=s-SPID,Date
					context.write(new IntWritable(closestCentroid),
							new Text("s-"+info));
					log.info("Mapper finished writing spid-start distance");
					//Write cost value with key=-1
					context.write(new IntWritable(-1), new Text(df.format(Math.pow(distance, 2))));	
					log.info("Mapper finished writing distance");
				}
			} 
		}
	}

	public static class CentroidReducer extends
			Reducer<IntWritable, Text, Text, Text> {

		public String KEY = "k";
		public String INFO = "s";
		public int NUM_POINTS = 96;
		public ArrayList<double[]> INIT = new ArrayList<double[]>();
		public Log log=LogFactory.getLog(CentroidReducer.class);

		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {

			// Get the centroids and keep them in memory
			// Uncomment for amazon //
			//String uriStr = "s3n://energydata/output/";
			String uriStr = "./output";
			//log.info("Reducer started fs-get centroid file");
			
			URI uri = URI.create(uriStr);
			FileSystem fs = FileSystem.get(uri, context.getConfiguration());	

			//FileSystem fs = FileSystem.get(context.getConfiguration());
			Path cFile = new Path(context.getConfiguration().get(CFILE));
			
			int  numOfClusters = Integer.valueOf(context.getConfiguration().get(NUMOFCLUSTERS));
			DataInputStream d = new DataInputStream(fs.open(cFile));
			BufferedReader reader = new BufferedReader(new InputStreamReader(d));
			String line;
			int centroidLength=0;
			log.info("Reducer started reading centroids");
			while ((line = reader.readLine()) != null) {
				if (!line.startsWith("C") && !line.startsWith("s")&& !line.startsWith("w")&& centroidLength !=numOfClusters) { 
					centroidLength++;
					//System.out.println(centroidLength);
					INIT.add(parsePoint(line));
				}
			}
			log.info("Reducer finished reading centroids");
			reader.close();
		}

		@Override
		protected void reduce(IntWritable key,
				Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			
			//System.out.println("Reducer key: " + key.get());

			ArrayList<String> infoList = new ArrayList<String>();
			if (key.get() == -1) { //we are reading the cost
				double cost = 0;
				int no = 0;
				// Get average for all dimensions and cost too !
				for (Text str : values) {
						no ++;
						cost = cost + Double.parseDouble(str.toString());
				}
				context.write(new Text("C" + "-" + key.toString()), new Text(df.format(cost)));				
			} else {
				double[] new_centroid = new double[NUM_POINTS];//new double[INIT.get(0).length];
				
				ArrayList<double[]> cluster_members = new ArrayList<double[]>();
				
				int count = 0;
				
				// double cost = 0;
				// Get average for all dimensions and cost too !

				log.info("Reducer started calculating" + Integer.toString(count));	
				for (Text str : values) {
					String input = str.toString();
					if (!input.startsWith("s")) {//if string has comma it is spid & date	
						double[] point   = new double[NUM_POINTS];
						String[] tokens = input.split(" "); 
						for (int i=0; i<NUM_POINTS;i++){
						     point[i]=Double.parseDouble(tokens[i]);
						     //average[i] =  average[i]+point[i];
						}
						cluster_members.add(point);
					    count++;
					}else { //just save the spid & date
						infoList.add(str.toString());
					}
				}

				// New centroid at center of mass for this cluster
				SimpleMatrix member_points = new SimpleMatrix(count,NUM_POINTS);
				for (int i=0;i<count;i++){
					double[] curPoint = cluster_members.get(i);
					System.out.print("\n");
					member_points.setRow(i, 0, cluster_members.get(i));
				}
				SimpleMatrix cur_center    = new SimpleMatrix(1,NUM_POINTS, true,INIT.get(key.get()));
				new_centroid = calculate_centroid(member_points, cur_center, count);
				log.info("Reducer finished calculating" + Integer.toString(count));

				// Emit new centroid
				String result = longArrayToString(new_centroid); 
				//log.info("Reducer started writing" + Integer.toString(count));
				context.write(new Text(KEY + "-" + key.toString()), new Text(
						result));
				//log.info("Reducer wrote key" + Integer.toString(count));
				context.write(new Text(INFO + "-" + key.toString()), new Text(
						infoList.toString()));
				log.info("Reducer ended writing" + Integer.toString(count));
			}
		}
	}

	public static void main(String[] args) throws Exception {		
		//Run as inputDir outputDir centroidDir no_of_iters max_clusters stepSize 
		//./ output/ ./energy-c1_norm.txt  2 10 10

		long startTime = System.currentTimeMillis();   //Measure elapsed time
		Configuration conf = new Configuration();

		//FileSystem fs = FileSystem.get(conf);
		String inputDir = args[0];
		//String opDirBase = args[1];
		String initCentroidsPath = args[2];
		int no_of_iters = 2;//Integer.valueOf(args[3]);  // Ucomment for amazon 
		int max_clusters = Integer.valueOf(args[4]);
		int stepSize=Integer.valueOf(args[5]);
		int min_clusters = 10;

		String uriStr = inputDir;
		URI uri = URI.create(uriStr);
		FileSystem fs = FileSystem.get(uri, conf);   
		//System.out.println("Working directory:"+fs.getWorkingDirectory().toString());
		String inputFiles = "";
		String baseFileName    = uriStr+"electric_interval_data_long_part";
		String suffix = "_96.txt";
		int numOfFiles = 10;
	    //*** uncomment this for hadoop		
		for (int i=1;i<=numOfFiles;i++){
			inputFiles +=baseFileName+ String.valueOf(i) +suffix+",";
		}
		/*** uncomment for amazon ***/
		//inputFiles = inputFiles.substring(0, inputFiles.length()-1);
		//inputFiles = uriStr+ "electric_interval_data_long_part1_and2_96.txt";
		inputFiles ="./input-only/test_shapes_100k.csv";
		//inputFiles ="./input-only/test_10.csv";
		//inputFiles = "./input-only/electric_interval_data_long_part1_and2_96.txt";


		//int j=50;
		String cDir = "";
		//for (int j=min_clusters; j<=max_clusters;j+=stepSize){  //Number of clusters
			int j=100; //Uncomment for amazon
			String opDirBase = args[1]+String.valueOf(j);
			conf.set(NUMOFCLUSTERS, String.valueOf(j));
	      	//System.out.println(conf.get(NUMOFCLUSTERS));
			System.out.println("# of Clusters:" + j
					+ "===========================================");
			for (int i = 0; i < no_of_iters; i++) {
				System.out.println("Iteration :" + i
						+ "===========================================");
				// Output dir in HDFS for this iteration
				String outputDir = opDirBase + "/" + i;
				// System.out.println("outputDir "+i+" :"+outputDir);
				String inPath = initCentroidsPath;

				// Merge o/p from previous job for jobs after the init run is
				// complete
				conf.set(NUMITERATIONS, String.valueOf(i));
				if (i > 0) {
					cDir = opDirBase + "/" + (i - 1);
					// System.out.println("cDir "+i+" :"+cDir);
					Path inDir = new Path(cDir);
					inPath = opDirBase + "/c/" + i + "-centroid.txt";
					Path newCentroid = new Path(inPath);
		            fs.delete(newCentroid, true); //delete new centroid path if it already exists
					// Centroid file name for this iteration
					//FileUtil.copyMerge(fs, inDir, fs, new Path(inPath), false,conf, "");
		            FileUtil.copyMerge(fs, inDir, fs, newCentroid, false,conf, "");

				}

				// Set centroid path
				conf.set(CFILE, inPath);
				// System.out.println(conf.get(CFILE));

				// Job Params
				Job job = new Job(conf, "KMeans");

				job.setJarByClass(edu.stanford.cs246.hw2.KSC.class);

				job.setMapperClass(CentroidMapper.class);
				job.setReducerClass(CentroidReducer.class);

				job.setMapOutputKeyClass(IntWritable.class);
				job.setOutputKeyClass(Text.class);
				job.setOutputValueClass(Text.class);

				//FileInputFormat.addInputPath(job, new Path(inputDir));
				//Change to have multiple inputs because input paths because energy data is patitioned into 10 txt files

				//System.out.println("input Files:"+inputFiles);

				//***uncomment this for multiple files
				FileInputFormat.addInputPaths(job, inputFiles);	

				Path outputPath = new Path(outputDir);
	            fs.delete(outputPath, true); //delete output path if it already exists
				//FileOutputFormat.setOutputPath(job, new Path(outputDir));
	            FileOutputFormat.setOutputPath(job, outputPath);
			    job.waitForCompletion(true);
			    //System.out.println("output Dir:" + outputPath);
			  // }
			}
			long stopTime = System.currentTimeMillis();
			long elapsedTime = stopTime - startTime;
		    //System.out.println(elapsedTime);
	}

	public static double[] parsePoint(String input) {
		double[]point = new double[96];
		int length = point.length;
		
		if (input.startsWith("k")) {
			String[] tk = input.split("\t");
			input = tk[1];
			String[] tokens = input.split(" ");
			//double[] point = new double[tokens.length];	
			//int length = point.length;
			for (int i = 0; i < length; i++) {
				point[i] = Double.parseDouble(tokens[i]);
			}

		} else{
			String[] tokens = input.split(",");
			if (tokens.length==1) {
				tokens = tokens[0].split(" ");
			}
			//double[] point = new double[96]; //[tokens.length-2-96];
			//int length = point.length;
			int offset=0;
			if (tokens[1].contains("-"))offset=2; //means we have spid & date as well
			//int daily_total=0;
			for (int i=0;i<length;i++){
				point[i]=Double.parseDouble(tokens[i+offset]);
				point[i]=point[i]/1000; //convert from wattH to kWh
			//	daily_total+=point[i];
			}
		/*	
			//Normalize
			if (daily_total!=0){
				for (int i=0;i<length;i++){
					point[i]=point[i]/daily_total;
				}
			}
	    */
		}

		return point;
	}

	public static String parseInfo(String input) {

		//String[] tk = input.split("\t");

		String[] line = input.split(",");//not sure if tab or space is better
		String info= line[0]+","+line[1];

		return info;

	}

	// Return the Euclidean distance between 2 points in r dimensions
	//shift centroid by 4 left and 4 right
	//Find optimal alpha for each shift
	//return minimum distance; 
	public static double distance(SimpleMatrix x, SimpleMatrix y) {
		double result = 0;
		double alpha  = 0;
		double minDistance = Double.MAX_VALUE;
		int num_shifts = 5;
		double pnorm=0;
		double curDistance = 0;
		int numPoints = 96;
		SimpleMatrix y_shift = new SimpleMatrix(1,numPoints,true,new double[numPoints]);
		
		for (int i=-num_shifts;i<=num_shifts;i++){
			y_shift.set(y);
			shift(y, y_shift,i);
			alpha = calculate_alpha(x, y_shift);   //alpha = x * y' / (y * y');
			if (x.elementSum()==0){
				pnorm=1;
			}else{
				pnorm=x.normF();
			}
			curDistance= x.minus(y_shift.scale(alpha)).normF()/pnorm;    //norm(x - alpha * y) / pnorm; pnorm=1 if point=0, else pnorm=norm(point);
			if (curDistance < minDistance) minDistance=curDistance;
		}	
		return minDistance;
	}
	
	public static SimpleMatrix distance_result (SimpleMatrix x, SimpleMatrix y) {
		double alpha  = 0;
		double minDistance = Double.MAX_VALUE;
		int num_shifts = 5;
		double pnorm=0;
		double curDistance = 0;
		int numPoints = 96;
		SimpleMatrix y_shift = new SimpleMatrix(1,numPoints,true,new double[numPoints]);
		SimpleMatrix min_y_shift = new SimpleMatrix(1,numPoints,true,new double[numPoints]);
		
		for (int i=-num_shifts;i<=num_shifts;i++){
			y_shift.set(y);
			shift(y, y_shift,i);
			alpha = calculate_alpha(x, y_shift);   //alpha = x * y' / (y * y');
			if (x.elementSum()==0){
				pnorm=1;
			}else{
				pnorm=x.normF();
			}
			curDistance= x.minus(y_shift.scale(alpha)).normF()/pnorm;    //norm(x - alpha * y) / pnorm; pnorm=1 if point=0, else pnorm=norm(point);
			if (curDistance < minDistance){
				minDistance=curDistance;
				min_y_shift.insertIntoThis(0, 0, y_shift);
			}
		}	
		return min_y_shift;
	}
	
	public static void shift(SimpleMatrix x, SimpleMatrix shifted, int shift_val){
		int numPoints = 96;		
		
		if (shift_val <0){
			double[] zeros=new double[-shift_val];
			shifted.setRow(0,numPoints+shift_val,zeros);   //yshift = [y(-shift + 1:end) zeros(1, -shift)];
			//SimpleMatrix inserted = x.extractMatrix(0,0,-shift_val,x.numCols()-1);
			//shifted.insertIntoThis(0,0,  x.extractMatrix(0, 0, -shift_val-1, x.END));
			for (int i=-shift_val;i<=numPoints-1;i++){ //do it manual way cause above didn't work!
				shifted.set(i+shift_val, x.get(i));
			}
		}else {
			if (shift_val !=0){
				double[] zeros=new double[shift_val];
				shifted.setRow(0,0,zeros);//yshift = [zeros(1,shift) y(1:end-shift) ];
				//shifted.insertIntoThis(0,1,x.extractMatrix(0, 0, shift_val, numPoints-shift_val-1));
				for (int i=shift_val;i<=numPoints-1;i++){ //do it manual way cause above didn't work!
					shifted.set(i, x.get(i-shift_val));
				}
			} 				
		}		
	}
	
	public static double calculate_alpha(SimpleMatrix x, SimpleMatrix y){
		double alpha=0;
		alpha = (x.mult(y.transpose()).get(0))/(y.mult(y.transpose()).get(0));//alpha = x * y' / (y * y');  B/A = (A'\B')'
		return alpha;
	}
	
	//Calculate new centroid for KSC
	public static double[] calculate_centroid(SimpleMatrix points, SimpleMatrix cur_center, int numRows) {
		int numPoints = 96;
		double [] norms_inv = new double[numRows];
		//double [] vector_norms = new double[numRows];
		
		double [] centroid = new double[numPoints];
		for (int i=0;i<numRows;i++){  //points_shifted=get shifted versions of all points;
			SimpleMatrix vector = points.extractVector(true, i);
			vector=distance_result(cur_center,vector);
			points.insertIntoThis(i,0,vector);
			//vector_norms[i] = vector.normF();
			double norm = vector.normF();
			
/*** Look at this again cause norm should never be zero. Might be something wrong ***/			
			if (norm!=0){
				norms_inv[i] = Math.pow(norm,-1);//Math.pow(vector.normF(),-1);
			}else{
				norms_inv[i] = 1;
			}
/****/				
		}
		
		SimpleMatrix points_norm_inv= new SimpleMatrix(numRows,numPoints);    //points.elementMult(points);		
		double[] repmat_norm_inv=new double[numPoints];
		for (int i=0;i<numRows;i++){
			for (int j=0;j<numPoints;j++){
				repmat_norm_inv[j]= norms_inv[i];  //norm_value;
			}
			points_norm_inv.setRow(i, 0, repmat_norm_inv); //SimpleMatrix a_norm = repmat(sqrt(sum(a.^2,2)), [1 size(a,2)]);
		}
	
		SimpleMatrix b= points.elementMult(points_norm_inv);
		SimpleMatrix M= b.transpose().mult(b).minus(SimpleMatrix.identity(numPoints).scale(numRows));
		
/****/		
		System.out.println("Matrix Size: " + "Rows: "+ numRows + " Cols "+ numPoints);
		System.out.println("M-Matrix Size: " + "Rows: "+ M.numRows() + " Cols "+ M.numCols());
		System.out.println("B-Matrix Size: " + "Rows: "+ b.numRows() + " Cols "+ b.numCols());
	
		
/*****		
		//Write Inv Norm matrix out to text file to look at values & try eigenvalue decomposition in matlab
		//Write to file
		 try{
			 String file_name = "vector-norm-test_matrix_" + String.valueOf(numRows) + ".txt";
			 FileWriter file = new FileWriter(file_name);
			 BufferedWriter out = new BufferedWriter (file);
			 for (int i=0; i<numRows; i++){
				 out.append(String.valueOf(vector_norms[i]));
				 out.append("\n");
				 //System.out.println(text);
				 //out.write(text);
			 }
		 out.close();
	     }catch (IOException e){
             System.out.println(e.getMessage());
	     }	
	        
	
		//Write Inv Norm matrix out to text file to look at values & try eigenvalue decomposition in matlab
		double[][] norms = new double[numRows][numPoints];
		for (int i=0; i<numRows; i++){	
			for (int j=0;j<numPoints;j++){
				norms[i][j] = b.get(i,j);
			}
		}
		//Write to file
		 try{
			 String file_name = "Norm-test_matrix_" + String.valueOf(numRows) + ".txt";
			 FileWriter file = new FileWriter(file_name);
			 BufferedWriter out = new BufferedWriter (file);
			 for (int i=0; i<numRows; i++){
				for (int j=0;j<numPoints;j++){
				 out.append(String.valueOf(norms[i][j]));
				 out.append(",");
				}
				 out.append("\n");
				 //System.out.println(text);
				 //out.write(text);
			 }
			 out.close();
	     }catch (IOException e){
             System.out.println(e.getMessage());
	     }	
	        
		
		//Write M matrix out to text file to look at values & try eigenvalue decomposition in matlab
		double[][] contents = new double[numRows][numPoints];
		for (int i=0; i<numRows; i++){	
			for (int j=0;j<numPoints;j++){
				contents[i][j] = b.get(i,j);
			}
		}
		//Write to file
		 try{
			 String file_name = "B-test_matrix_" + String.valueOf(numRows) + ".txt";
			 FileWriter file = new FileWriter(file_name);
			 BufferedWriter out = new BufferedWriter (file);
			 for (int i=0; i<numRows; i++){
				for (int j=0;j<numPoints;j++){
				 out.append(String.valueOf(contents[i][j]));
				 out.append(",");
				}
				 out.append("\n");
				 //System.out.println(text);
				 //out.write(text);
			 }
			 out.close();
	     }catch (IOException e){
             System.out.println(e.getMessage());
	     }	
	        
	
		//Write M matrix out to text file to look at values & try eigenvalue decomposition in matlab
		double[][] contents = new double[numPoints][numPoints];
		for (int i=0; i<numPoints; i++){	
			for (int j=0;j<numPoints;j++){
				contents[i][j] = M.get(i,j);
			}
		}
		//Write to file
		 try{
			 String file_name = "test_matrix_" + String.valueOf(numRows) + ".txt";
			 FileWriter file = new FileWriter(file_name);
			 BufferedWriter out = new BufferedWriter (file);
			 for (int i=0; i<numPoints; i++){
				for (int j=0;j<numPoints;j++){
				 out.append(String.valueOf(contents[i][j]));
				 out.append(" ");
				}
				 out.append("\n");
				 //System.out.println(text);
				 //out.write(text);
			 }
			 out.append("\n");
			 out.close();
	     }catch (IOException e){
             System.out.println(e.getMessage());
	     }	
	        
****/	
		SimpleEVD eigenDecomp = M.eig();
		SimpleMatrix centroid_matrix=eigenDecomp.getEigenVector(eigenDecomp.getIndexMin());

		double centroid_sum = centroid_matrix.elementSum();
		
		if (centroid_sum >=0){
			for (int i=0;i<numPoints;i++){
				centroid[i]=centroid_matrix.get(i);
			}
		}else{
			for (int i=0;i<numPoints;i++){
				centroid[i]=-centroid_matrix.get(i);
			}
		}

		return centroid;
	}

	public static String longArrayToString(double[] average) {
		String result = new String();
		for (int i = 0; i < average.length; i++) {
			result = result + df.format(average[i]);
			if (i != average.length) {
				result = result + " ";
			}
		}
		return result;
	}

	public static class DoubleArrayWritable extends ArrayWritable {
		public DoubleArrayWritable() {
			super(DoubleWritable.class);
		}

		public DoubleArrayWritable(DoubleWritable[] values) {
			super(DoubleWritable.class, values);
		}
	}

}

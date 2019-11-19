# [Report] Introduction to Big Data Assignment №2.
### Stream Processing with Spark

---

### The team and responsibilities of team members, commit history
[Link to our GitHub Repository](https://github.com/HiGal/SenAnalysis).

**Arina Kuznetsova:** report, stream reading, data labeling
**Farit Galeev:** models creation and analysis
**Mikhail Tkachenko:** stream reading, wordcount
**Vladislav Savchuk:** setting up filters, connecting stream to models, wordcount, tokens cleaning, model training and tuning

---

### Description of the problem
The goal of this assignment is to perform Sentiment Analysis and define the emotional coloring of a stream data from Twitter using Apache Spark, Scala and Machine Learning approaches. 

Social Sentiment Analysis problem statement is to analyze the sentiment of social media content, like tweets. The algorithm takes a string, and returns the sentiment rating for the “*positive*” or “*negative*”.

From Twitter, the sentiment analysis scope (*data based on which the analysis is performed*) is **sentence level** as it usualy obtains the sentiment of *a single sentence*.

In addition to sentiment analysis, we need to show top 10 most popular tweets topics, excluding the most popular non-meaningful words (*e.g. prepositions, conjunctions, pronouns*).

---

### Our solution for the problem
### a. Algorithm. Working schema of program
![](https://i.imgur.com/i9wgRja.png)




### Explanation of the algorithm
### Word Count Part
1. In word count, we take lines which come from the Stream, and then start to work with them.
    a.  First, we remove html tags from the line that came
    ```
    val words = lines.map(line => "&(lt)?(gt)?(amp)?(quot)?;".r.replaceAllIn(line, ""))
    ```
    b. Then, we remove username, as they should not be         considered as topics
    `.map(line => "@[a-zA-Z0-9_]+".r.replaceAllIn(line, ""))`
    c. Split the line into words
    `.flatMap(line => line.split("\\W"))`
    d. To avoid considereation of words that start from capital letter as a new ones, make all of the words lowercase
    `.map(word => regex.replaceAllIn(word.trim.toLowerCase, ""))`
    e. Filtering and removing empty words (they may appera in case of multiple spaces) and stop words (shouldn't be considered a topic)
    ```
    .filter(word => !word.isEmpty)
    .filter(word => !stopwords.contains(word)
    ```
    f. Mapping a word to tuple with word as key and 1 as value
       `.map(word => (word, 1))`
    g. Count the words by reducing by key within a certain time period (6 hours in our example)
    `.reduceByKeyAndWindow(_ + _, Seconds(21600))`
    h. Sort the words
    ```
    .map(tuple => (tuple._2, tuple._1))
    .transform(rdd => rdd.sortByKey(false))
    ```
2. After finishing the classifications, we are saving files containing words and their appearance in tweets. Batch time windows is 6 hours and tweet period is 1 minute, so the file will be written every minute, but the data will be accumulated for 6 hours. The files represent the most recent popular topics. 
---
### Sentiment Analysis Part
1. The input of the algorithm is a stream, so first strings from it should be transformed into **DataFrame** for further processing.
2. Then, the DataFrame goes through ***preprocessing piplene***:
    
    **Preprocessing**
    
     a. **Cleaner**. Our own implementation of transformer, so that we can use it in pipeline (can be found in scala source directory in package `utilities`). It replaces all the substrings matching the         Regular Expressions (`"@[a-zA-Z0-9_]+"`,` "&(lt)?(gt)?           (amp)?(quot)?;"`) to remove the nickname and html tags           from the tweet. We are doing it to avoid processing of words         and symbols that are not related to the sentence that           the algorithm will analyze.
     
     b. **Tokenizer**. It fetches all the words (except removed         ones) from the initial sentence.
     
     c. **Stop Words Remover**. It removes articles,                     prepositions, conjunctions, pronouns, as they are not           semantically important to define the sentiment.
     
     **Feature Extraction**
     
     d. **Word2Vec**. From words, we need to get representation         of them, which can be understand by the model. So,               Word2Vec vectorizes words, that are semantically                 important.
     
     e. **Normalizer**. Brings the values to the same scale. 
    
3. **Classification**. We used 4 models for classification: Logistic Regression, Multilayer Perceptron, Linear SVC and Random Forest.
4. **Output**. Several CSV-files with data in `Date  Time, Sentence, Class` format, than they are easily accumulated to a single CSV file and saved in HDFS.

### b. A code
[Link to our GitHub Repository](https://github.com/HiGal/SenAnalysis).
Description of the solution is described above and comments in code will help you to understand

### c. Our improvements and suggestions. Comparison of our solution to others
We have used 4 classification models: **Logistic regression**, **Random Forest** classifier, **Multilayer perceptron** classifier and **Linear SVC**. We have also tried to develop a CNN network, but it required too much memory to run. The biggest accuracy on validation dataset is reached by logistic regression: ~73%. Logreg also gives best accuracy on stream: ~80%. Results of other models can be found further in the report.

Unfortunately, we cannot test CNN text classifier model due to a lot of parameters in the model and memory bound. Firstly, we used GloVe for word embeddings that takes a lot of place in RAM, secondly,the model with high numbers of parameters which is hard to debug on cluster. This model by expectation should outperform other models with F1-score ~0.89.

Summary of the model
![](https://i.imgur.com/dGk30J6.png)

### d. How and on what data training and testing was carried out
Data training and testing was performed on [this dataset](https://www.kaggle.com/c/twitter-sentiment-analysis2/data). This dataset consists of tweets and their class - either *negative* - 0 - or *positive* - 1. 

How data looks before Preprocessing
```
+------+---------+--------------------+
|ItemID|Sentiment|       SentimentText|
+------+---------+--------------------+
|     1|        0|                 ...|
|     2|        0|                 ...|
|     3|        1|              omg...|
|     4|        0|          .. Omga...|
|     5|        0|         i think ...|
|     6|        0|         or i jus...|
|     7|        1|       Juuuuuuuuu...|
|     8|        0|       Sunny Agai...|
|     9|        1|      handed in m...|
|    10|        1|      hmmmm.... i...|
|    11|        0|      I must thin...|
|    12|        1|      thanks to a...|
|    13|        0|      this weeken...|
|    14|        0|     jb isnt show...|
|    15|        0|     ok thats it ...|
|    16|        0|    &lt;-------- ...|
|    17|        0|    awhhe man.......|
|    18|        1|    Feeling stran...|
|    19|        0|    HUGE roll of ...|
|    20|        0|    I just cut my...|
+------+---------+--------------------+
```

How data looks after Preprocessing Pipeline
```
+------+---------+--------------------+--------------------+
|ItemID|Sentiment|       SentimentText|           normedW2V|
+------+---------+--------------------+--------------------+
|     1|        0|                 ...|[0.08277315133301...|
|     2|        0|                 ...|[0.06367116317163...|
|     3|        1|              omg...|[0.04960844079045...|
|     4|        0|          .. Omga...|[0.02645690954498...|
|     5|        0|         i think ...|[0.06703980878293...|
|     6|        0|         or i jus...|[-0.0354196422615...|
|     7|        1|       Juuuuuuuuu...|[-0.0434029851201...|
|     8|        0|       Sunny Agai...|[0.08388142929486...|
|     9|        1|      handed in m...|[-0.0074605287753...|
|    10|        1|      hmmmm.... i...|[0.09769357629731...|
|    11|        0|      I must thin...|[-0.0657444270670...|
|    12|        1|      thanks to a...|[0.11535828633919...|
|    13|        0|      this weeken...|[-0.0339398082716...|
|    14|        0|     jb isnt show...|[0.00918934463151...|
|    15|        0|     ok thats it ...|[0.07776135736402...|
|    16|        0|    -------- This...|[0.03118812509376...|
|    17|        0|    awhhe man.......|[0.06663149978470...|
|    18|        1|    Feeling stran...|[-9.0368754491392...|
|    19|        0|    HUGE roll of ...|[0.01143927921501...|
|    20|        0|    I just cut my...|[0.03110880174509...|
+------+---------+--------------------+--------------------+
```

For training **train.csv** and **test.csv** were used. For testing we used **train.csv** with train-validation split. For tuning we used a 5-fold Cross Validation on the train data. Below, when we will talk about models performance, the process of model training will be described more thoroughly.


![](https://i.imgur.com/Mk20bfi.png)




### An example of the work of your program with a description of the outputs and the decisions made

#### Examples of first strings of ouput files for different models

| Date     | Time     | Text     |Linear SVC Label|
| -------- | -------- | -------- |--------|
| 11-19    | 11:22    | @BlokesLib Night night sweetie! Have a great one!     | 1|
|11-19|11:23|	@BlokesLib oh...except when i have to refuel from drums...that sucks|0|


| Date     | Time     | Text     |LogReg Label|
| -------- | -------- | -------- |--------|
| 11-19    | 11:22    | @BlokesLib Night night sweetie! Have a great one!     | 1|
|11-19|11:23|	@BlokesLib oh...except when i have to refuel from drums...that sucks|0|


| Date     | Time     | Text     |Random Forest Label|
| -------- | -------- | -------- |--------|
| 11-19    | 11:22    | @BlokesLib Night night sweetie! Have a great one!     | 1|
|11-19|11:23|	@BlokesLib oh...except when i have to refuel from drums...that sucks|0|


| Date     | Time     | Text     | Multilayer Perceptron Label|
| -------- | -------- | -------- |--------|
| 11-19    | 11:22    | @BlokesLib Night night sweetie! Have a great one!     | 1|
|11-19|11:23|	@BlokesLib oh...except when i have to refuel from drums...that sucks|0|

The output of the models in stored in HDFS folder of our group - **/user/sharpei/output** with directory for every model. So the output structure is:
**/user/sharpei/output/logistic/out.csv** -- Output of Logistic Model
**/user/sharpei/output/random_forest/out.csv** -- Output of Random Forest Model
**/user/sharpei/output/svc/out.csv** -- Output of Linear SVC Model
**/user/sharpei/output/perceptron/out.csv** -- Output of Multilayer Perceptron Model
**/user/sharpei/output/topics/out.csv** -- Most popular topics with count (word, count)



---

### Selected models and their results. Classifiers analysis. Explanaition of results.


|Model| Accuracy on validation set |F1 Score on validation set| Accuracy on Streaming Data| F1 Score on Streaming Data | 
| --------------------|------|------|----|--------|
|Logistic Regression  |0.7381|0.6862|0.80| 0.8391 |
|Multilayer Perceptron|0.7370|0.6854|0.74| 0.8391 |
|Random Forest        |0.6766|0.5127|0.78| 0.8553 |
|Linear SVC           |0.7361|0.6849|0.76| 0.8391 |



We have used 4 models, that are widely used in Machine Learning. The results of their work is shown above.

As we know Neural Networks outperform Random Forests on image classification, from the high point of view text classification doesn't differ from image classfication, because firstly we make a word embedding and after that create a list of word embeddings for the sentence which gives us a 2D arrays as a grayscale image. It is because RFC is very sensetive to the preprocessing and to the number of features rather then Neural Networks.

The score between MLP, Logistic Regression and Linear SVC is the same because of the hyperparameters, but theoretically with a better hyperparameters MLP should beat Linear SVC and Linear SVC should beat Logistic Regression because it less sensitive to outliers then LogReg.

The code of training models is located in [TrainModels.scala](https://github.com/HiGal/SenAnalysis/blob/feature/io/src/main/scala/TrainModels.scala) file.For training, we use *train.csv*.
The Usage of the program:
```
Usage: TrainModels <dataset_folder> <model_name>

Possible models:
All require dataset train.csv in the dataset directory
	word2vec - word2vec model
All those below also require a pipeline pretrained model
	logistic - logistic regression
	perceptron - multilayer perceptron model
	svm - linear SVC model
	forest - Random Forest model
```
The output is then saved to `model_name.model` and will be loaded in the main algorithm (SentimentAnalyzer) for predioction.

Hyperparameter tuning is performed in [ParameterTuning.scala](https://github.com/HiGal/SenAnalysis/blob/feature/io/src/main/scala/ParameterTuning.scala). For finding the best parameters with the highest accuracy for a certain model, we have used **GridSearch** with 5-fold Cross Validation on *train.csv*, it will then output the best models accuracy and it's parameters
```
Usage: ParameterTuning <model_name>
Available models:
	logistic - Logistic Regression model
	random_forest - Random Forest model
```
***Note***: after finding the best parameters, it is needed to manually insert them into models' code.

According to accuracy and F1-score on validation dataset, the top of the models is the following:
    1. Logistic Regression
    2. Multilayer Perceptron
    3. Linear SVC
    4. Random Forest
    
According to accuracy and F1-score on real data:
    1. Random Forest / Logistic Regression
    2. Linear SVC
    3. Random Forest

    
**Set Size**: 100 tweets
[**Tweets, true and predicted values, presicion, recall, F1-score**](https://docs.google.com/spreadsheets/d/1FWTp9A8lMqkJUZ914TEjJiv20zqXqTK4ylZ-rcOSbK4/edit#gid=723827142)
    
**Overall**: The best model from our experiments, is Logistic Regression Model  as it shows the best performance on validation and stream data, and F1-score shows, that these accuracies are achieved without overfitting. So that, we know that on Image Classification CNN outperforms MLP and theoretically it should outperform it on Text Classifcation.

---

### Top 10 most popular tweet topic
![](https://i.imgur.com/jSEAxQT.png)

Some words like *m, ll* appeared after removing stop words (m -> I'm, ll -> I'll). Word *u* appeared as a shortened version of the word *you*. 

Such word have appeared as a result of not full list of Stop Words.

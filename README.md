## Finding worth of questions of Jeopardy
- Dataset Link: https://www.kaggle.com/tunguz/200000-jeopardy-questions
- Info about the dataset:
    -- The the csv is has many columns, our interest is the " Questions" and the " Value" columns
   
   
### PreProcessing:
-The preProcess.py file does the following this as per the latest commit:
  -- Convert strings in form of "$4,000" to intergers in the " Value" columns
  -- Does common nlp preProcessing (removing punctuations, etc) on the Questions column
  --- Note: here we are reading from a text file, but first time you run, you must create the file:
          
    with open('listfile.txt', 'w') as filehandle:
        for listitem in Questions:
            filehandle.write('%s\n' % listitem)
            
            
            
  -- Does word2vec on Google's word2vec300.negetive
  -- Performs weighted multiplication using tf-idf on the Questions
  -- Get the mean of the word vectors of the sentences and make the training set
  
 
### training:
-Have used decsion trees here for training, but using any ensemble methods gives the same accuracy as the model is underfitting


### Future aspects:
- Can train the model on LSTM
- The current implementation crashes the computer when saving the numpy array of the training set, try to work around that.


### Requirements
- See requrements.txt

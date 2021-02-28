


> Written with [StackEdit](https://stackedit.io/).
> **Assignment Approach**

### Day 0: Date 24/02/2020

- I got a response from Wysa. They were hiring for the position of intern. For judging my technical skills I was given an assignment.

- There were two options to choose from:

- Recommendation of books on Goodreads
- Predicting the worth of question from Jeopardy

- Found out what is Jeopardy and defined the problem

- I chose the latter one. Saw the data on the zeroth day and started reading about NLP as much as i could, because of my lack of experience in NLP.

### Day 1: Date 25/02/2020

- Started reading on how to vectorise words and the basics of NLP. Going down to every subtleties , as i would like to do whenever i start a project, was not an option because of the time constraint.

- Downloaded the dataset and started exploring, the csv file

- Out of the feature set, the target feature was named &quot; Value&quot; and contained data in form of string like &quot;$4,000&quot;

- Started the preProcessing stage:

- Converted the data in the &#39;Value&#39; column to integer, changed the name to &quot;ValueNum&quot;
- There were more than 120 different values in the &quot; Value&quot; column. Jeopardy doesn&#39;t questions whose value has this sort of range
- Had to bin this value to:
  - The Nearest 100 if value \&lt; 1000
  - The Nearest 1000 if value \&lt; 10000
  - The Nearest 10000 if value \&lt; 100000

- Then started preProcessing on the Questions:

- Removed Punctations
- Removed accented characters from text, e.g. café
- Removed extra whitespaces from text
- Expanded shortened words, e.g. don&#39;t to do not

Started stumbling on vectorization.

- Now as per my then limited knowledge, I had to vectorize the Input data

- Experimented around bagOfWords but was not able to achieve higher accuracy, because the corpus was too small.

### Day 2:  26/02/2020

Did some experimentation around BERT but got a several setbacks

Setbacks :

- Didn&#39;t had prior knowledge around bert architecture, spend majority of the day trying to make sense of it
- Came up with a working pipeline around BERT and spent majority of night around it
- My local machine was constantly throwing CUDA memory error, tried to debug this.
- Transferred the code to Kaggle but yet the accuracy was too low as big enough corpus was absent

### Day3 27/03/2020

- Consulted one of my mentors (Mr Sohum Mehta) about the problem. His suggestion of solving the corpus problem was to refer GLOVE project by Stanford

- Made Glove the part of the pipeline:

- Converted the glove.6b50d model to dictionary of labels

- Made vector representation of the Questions:

- For each question made a vector of wordVectors, these wordVectors were vector representations of words in the sentences

- Took mean of the vector to get a scalar representing the sentence

- Made Decision tree model to train these vectors upon, got poor accuracy:

- On investigation it was found that the questions had many proper nouns as such, and GLOVE didn&#39;t had embeddings for that

- Tried a new vectorizer called &quot;GoogleNews-vectors-negative300&quot; which is trained on Google News

- This had comparatively less missing values, handled them and trained the model
### Day 4 28/7/2021

- The accuracy was pretty bad, came up with an idea of multiplying weights of the words in the corpus with the representation in the model.


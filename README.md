# Text Classification
The aim of this project is to classify richly formatted unstructured textual data (datasheets) of electronic components. This is a part of a datasheet scraping project. 

## Dependencies
for pdftotext: 
https://pypi.org/project/pdftotext/

## Usage
Run 
```bash
main.py
``` 
for processing the documents, training the model, and validate its performance.

We can choose between the following models:


**Desicion Tree** \
**Naive Bayes** \
**Random Forest** \
**Logistic Regression** \
**SGD** (SVM with Gradient Descent) \
**SVM** \
**Rocchio** \
**Extra Trees** \
**fastText** \
**DistilBERT** \
**Roberta** 

### Clustering
We can run 
```bash
src/clustering.py
``` 
to cluster our documents using the k-means algorithm and visualize the results using dimensionality reduction.

### Topic Modeling

We can run
```bash
src/topic_modeling.py
```

to apply lda for topic modeling on our documents. The results are saved in a HTML file.

### Visualization

We can run
```bash
src/visualize_data.py
```
and 
```bash
src/word_cloud.py
```
to visualize our data

### Dimensionality Reduction
We can run 
```bash
src/dimensionality_reduction.py
```
to perform various dimensionality reduction techniques (lsa, nmf, kbest(chi2)) and test how they affect the classifiers' performance



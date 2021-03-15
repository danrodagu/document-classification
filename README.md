# Document Classification

## Justification
This Machine Learning project is meant to compare two different algorithms to classify documents. For this purpose, Naive Bayes and kNN algorithms have been used.

The **Naive Bayes** algorithm is based on Bayes' theorem and its classifier assumes that the presence or absence of a particular feature is not related to the presence or absence of any other feature, given the variable category.

On the other hand, the **kNN** algorithm estimates the a posteriori probability that an element belongs to a certain category based on the information provided by the training set, classifying it into the most frequent class to which its K nearest neighbors belong.

## How to execute

### *What do you need to be able to run the program correctly?*

- Python 3.6.0
- NLTK Natural Language Toolkit

If you do not have NLTK downloaded, a file “nltk_download.py” is attached to the project Workspace. To download it, you will only need to run this file with its corresponding IDE.

***Note:** If any error is thrown when executing the “nltk_download.py” file, it will be necessary to install NLTK previously following the steps from this source: “http://www.nltk.org/install.html”*

### *How to add a category and its keywords?*

1. Access the training categories path *“…/entrenamiento/categorias”*.
2. Create a “.txt” file whose title (without accents) must be the name of the category you want to add.
3. Each keyword you want to add must be written on a different line in the document.

***Note:** if the name of the category to be added contains a number, this must be omitted or an alternative identifying name without numbers must be determined for said category.*

### *How to add documents to the training set?*

1. Access the training texts path *“…/entrenamiento/textos”*.
2. Create a “.txt” file with the format *“{category}{CategoryTextid}.txt”*. Example: for the category “football” the texts to add will be “football1.txt”, “football2.txt”, etc.
3. The file must contain the text of the document that we want to add as training text.

### *How to add additional documents to test the classifiers?***

1. Access the additionals path *“…/adicionales”*.
2. Create a “.txt” file with the format *“additional{category}{additionalCategoryTextid}.txt”*. Example: for the category “football” the additional texts to add will be “additionalfootball1.txt”, “additionalfootball2.txt”, etc.
3. The file must contain the text of the document that we want to add as additional text.

### *How to execute the training and classification processes?*

1. Open the file “metodos.py” found in the project path.
2. To execute the training processes we will go to the end of the document and only leave the line corresponding to the training we want to execute uncommented. Example: entrenamiento_knn().
3. To execute the classification processes we will go to the end of the document and only leave the line corresponding to the classifier that we want to execute uncommented. Example: clasificar_documento_knn(“additionaltennis2.txt”, 7).

***Note:** the classification results will be displayed in the terminal.*

## Notes
This project is meant to be executed with texts in Spanish.
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords


train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
enc = OneHotEncoder(handle_unknown='ignore')
le = preprocessing.LabelEncoder()

X_train = [str(x) for x in train_set['title']]




trainDf = pd.DataFrame(train_set, columns=["title", "label"])
testDf = pd.DataFrame(test_set, columns=["title", "label"])
# train_categories = trainDf["label"]
# test_categories = testDf["label"]

# print(trainDf.head)




wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))

def tokenize_lemma_stopwords(text):
    text = str(text).replace("\n", " ")
    # split string into words (tokens)
    tokens = nltk.tokenize.word_tokenize(text.lower())
    # keep strings with only alphabets
    tokens = [t for t in tokens if t.isalpha()]
    # put words into base form
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] 
    tokens = [stemmer.stem(t) for t in tokens]
    # remove short words, they're probably not useful
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    cleanedText = " ".join(tokens)
    return cleanedText

def dataCleaning(df):
    data = df.copy()
    data["title"] =data["title"].apply(tokenize_lemma_stopwords)
    return data
cleanedTrainData = dataCleaning(trainDf)
cleanedTestData = dataCleaning(testDf)

# print(cleanedTrainData)





from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

vectorizer = TfidfVectorizer()
vectorised_train_documents = vectorizer.fit_transform(cleanedTrainData["title"])
vectorised_test_documents = vectorizer.transform(cleanedTestData["title"])

# from yellowbrick.text import FreqDistVisualizer
# features = vectorizer.get_feature_names()
# visualizer = FreqDistVisualizer(features=features, orient='v')
# visualizer.fit(vectorised_train_documents)
# visualizer.show()


# from yellowbrick.text import UMAPVisualizer

# umap = UMAPVisualizer(metric="cosine")
# umap.fit(vectorised_train_documents)
# umap.show()


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
Y_train = [str(x).split(';') for x in train_set['label']]
train_labels = mlb.fit_transform(Y_train)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()



Y_test = [str(x).split(';') for x in test_set['label']]
test_labels = mlb.transform(Y_test)





from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss

ModelsPerformance = {}

def metricsReport(modelName, test_labels, predictions):
    accuracy = accuracy_score(test_labels, predictions)

    macro_precision = precision_score(test_labels, predictions, average='macro')
    macro_recall = recall_score(test_labels, predictions, average='macro')
    macro_f1 = f1_score(test_labels, predictions, average='macro')

    micro_precision = precision_score(test_labels, predictions, average='micro')
    micro_recall = recall_score(test_labels, predictions, average='micro')
    micro_f1 = f1_score(test_labels, predictions, average='micro')
    hamLoss = hamming_loss(test_labels, predictions)
    print("------" + modelName + " Model Metrics-----")
    print("Accuracy: {:.4f}\nHamming Loss: {:.4f}\nPrecision:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nRecall:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nF1-measure:\n  - Macro: {:.4f}\n  - Micro: {:.4f}"\
          .format(accuracy, hamLoss, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1))
    ModelsPerformance[modelName] = micro_f1



from sklearn.ensemble import BaggingClassifier

bagClassifier = OneVsRestClassifier(BaggingClassifier(n_jobs=-1))
bagClassifier.fit(vectorised_train_documents, train_labels)
bagPreds = bagClassifier.predict(vectorised_test_documents)
metricsReport("Bagging", test_labels, bagPreds)

from sklearn.svm import LinearSVC

svmClassifier = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svmClassifier.fit(vectorised_train_documents, train_labels)

svmPreds = svmClassifier.predict(vectorised_test_documents)
metricsReport("SVC Sq. Hinge Loss", test_labels, svmPreds)




from sklearn.ensemble import GradientBoostingClassifier

boostClassifier = OneVsRestClassifier(GradientBoostingClassifier())
boostClassifier.fit(vectorised_train_documents, train_labels)
boostPreds = boostClassifier.predict(vectorised_test_documents)
metricsReport("Boosting", test_labels, boostPreds)




from sklearn.ensemble import RandomForestClassifier
rfClassifier = RandomForestClassifier(n_jobs=-1)
rfClassifier.fit(vectorised_train_documents, train_labels)
rfPreds = rfClassifier.predict(vectorised_test_documents)
metricsReport("Random Forest", test_labels, rfPreds)





from skmultilearn.problem_transform import LabelPowerset

powerSetSVC = LabelPowerset(LinearSVC())
powerSetSVC.fit(vectorised_train_documents, train_labels)

powerSetSVCPreds = powerSetSVC.predict(vectorised_test_documents)
metricsReport("Power Set SVC", test_labels, powerSetSVCPreds)









# from sklearn.preprocessing import MultiLabelBinarizer
# mlb = MultiLabelBinarizer()
# new_Y_train = mlb.fit_transform(Y_train)

# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer()


# # X_train = cv.fit_transform(X_train)

# print(X_train.shape)
# # print(mlb.inverse_transform(np.array([0 1 0 0 0 0 0 1 0])))




# # X_test = [str(x) for x in test_set['title']]
# Y_test = [str(x).split(';') for x in train_set['label']]

# # X_test = cv.transform(X_test) # WE NEED A BETTER WAY TO CREATE FEATURES FOR INPUT!!!!



# new_Y_test = mlb.transform(Y_test)

'''
from sklearn.ensemble import RandomForestClassifier
print(X_test.shape)

clf = RandomForestClassifier(max_depth=5, random_state=0)

# from sklearn.svm import SVC
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, new_Y_train)
Y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

# print(Y_pred)
# print(Y_test)

print(Y_pred[:10])
print(new_Y_test[:10])
'''
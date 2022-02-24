

import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):#enumerate 函数将数据转换为索引序列，同时列出数据和数据下标
                if i == 2:
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)#Counter函数用来统计词频
    #Remove non-words and single characters
    list_to_remove = dictionary.keys()#keys函数：以列表返回一个字典所有的键
    for item in list(list_to_remove):
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary
    
def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix
    
# Create a dictionary of words with its frequency

train_dir = 'lingspam_public\\lemm_stop\\train-mails'
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors per training mail and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)

# Training SVM and Naive bayes classifier and its variants

model1 = LinearSVC()
model2 = MultinomialNB()
model3 = RandomForestClassifier()
model4 = ExtraTreesClassifier()

model1.fit(train_matrix,train_labels)
model2.fit(train_matrix,train_labels)
model3.fit(train_matrix,train_labels)
model4.fit(train_matrix,train_labels)
# Test the unseen mails for Spam

test_dir = 'lingspam_public\\lemm_stop\\test-mails'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1

actual_result_1 = model1._predict_proba_lr(test_matrix)
actual_result_2 = model2.predict_proba(test_matrix)
actual_result_3 = model3.predict_proba(test_matrix)
actual_result_4 = model4.predict_proba(test_matrix)
result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)
result3 = model3.predict(test_matrix)
result4 = model4.predict(test_matrix)

actual_result_3=list(actual_result_3)
actual_result_4=list(actual_result_4)


ac_score_1 = metrics.accuracy_score(test_labels, result1)
ac_score_2 = metrics.accuracy_score(test_labels, result2)
ac_score_3 = metrics.accuracy_score(test_labels, result3)
ac_score_4 = metrics.accuracy_score(test_labels, result4)


print(actual_result_3)
print(actual_result_4)
print(ac_score_1)
print(ac_score_2)
print(ac_score_3)
print(ac_score_4)

print(confusion_matrix(test_labels,result1))
print(confusion_matrix(test_labels,result2))
print(confusion_matrix(test_labels, result3))
print(confusion_matrix(test_labels, result4))



from sklearn import svm
import pandas
import numpy as np
import csv
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification

#[A C E D G F I H K M L N Q P S R T W V Y]

def array_of_features_of_sequence(seq):

	feature1=ratio_amino_acid(seq)#its an array of size 20
	feature2=dipeptide_comp(seq)#all dipeptidebonds i.e. array of size 400
	features= merge(feature1,feature2)#420 size array

	return features

def ratio_amino_acid(seq):
	arr=[0 for i in range(20)] #[A C E D G F I H K M L N Q P S R T W V Y]
	for i in range(len(seq)):
		if(seq[i]=='A'):
			arr[0]+=1
		elif(seq[i]=='C'):
			arr[1]+=1
		elif(seq[i]=='E'):
			arr[2]+=1
		elif(seq[i]=='D'):
			arr[3]+=1
		elif(seq[i]=='G'):
			arr[4]+=1
		elif(seq[i]=='F'):
			arr[5]+=1
		elif(seq[i]=='I'):
			arr[6]+=1
		elif(seq[i]=='H'):
			arr[7]+=1
		elif(seq[i]=='K'):
			arr[8]+=1
		elif(seq[i]=='M'):
			arr[9]+=1
		elif(seq[i]=='L'):
			arr[10]+=1
		elif(seq[i]=='N'):
			arr[11]+=1
		elif(seq[i]=='Q'):
			arr[12]+=1
		elif(seq[i]=='P'):
			arr[13]+=1
		elif(seq[i]=='S'):
			arr[14]+=1
		elif(seq[i]=='R'):
			arr[15]+=1
		elif(seq[i]=='T'):
			arr[16]+=1
		elif(seq[i]=='W'):
			arr[17]+=1
		elif(seq[i]=='V'):
			arr[18]+=1
		elif(seq[i]=='Y'):
			arr[19]+=1

	#to get ratio we divide by length
	
	for i in range(20):
		arr[i]=(arr[i]*1.0)/(len(seq))

	return arr


def dipeptide_comp(seq):
	bonds_array=[0.0 for i in range(400)]

	for i in range(len(seq)-1):
		char1=seq[i]
		char2=seq[i+1]
		ind1=find_index(char1)
		ind2=find_index(char2)
		bonds_array[ind1*20+ind2]+=1.0
	
	for i in range(400):
		bonds_array[i]=(bonds_array[i]*1.0)/(len(seq)-1)
	return bonds_array

def find_index(char):
	total_amino_acids=['A', 'C', 'E' ,'D', 'G' ,'F' ,'I', 'H' ,'K' ,'M' ,'L' ,'N' ,'Q' ,'P' ,'S' ,'R' ,'T' ,'W' ,'V' ,'Y']
	for i in range(len(total_amino_acids)):
		if(char==total_amino_acids[i]):
			return i

def merge(arr1,arr2):
	arr=[]
	for i in range(len(arr1)):
		arr.append(arr1[i])
	for i in range(len(arr2)):
		arr.append(arr2[i])
	return arr

##main code
#storing data into pandas dataframes
train_data=pandas.read_csv('train.csv')
labels=train_data.iloc[:,[1]].values 
sequences=train_data.iloc[:,[2]].values 


features_array=[] #X
labels_arr=[] #Y
for i in range(len(labels)):
	labels_arr.append(labels[i][0])
for i in range(len(sequences)):
	seq=sequences[i][0]
	all_features=array_of_features_of_sequence(seq)
	features_array.append(all_features)
# print(features_array)
# print(labels_arr)

# Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(features_array, labels, test_size = 20, random_state = 100)
#defining a stump - random forest 
stump2 = DecisionTreeClassifier(max_depth = 1, splitter = "best", max_features = "sqrt")
ensemble2 = BaggingClassifier(base_estimator = stump2, n_estimators = 1000,
                             bootstrap = True)
#extra trees classifier
stump = DecisionTreeClassifier(max_depth = 1, splitter = "random", max_features = "sqrt")
ensemble = BaggingClassifier(base_estimator = stump, n_estimators = 1000,
                             bootstrap = False)

#Training Classifiers
stump.fit(X_train, np.ravel(y_train))
ensemble.fit(X_train, np.ravel(y_train))

# Making predictions
y_pred_stump = stump.predict(X_test)
y_pred_ensemble = ensemble.predict(X_test)


# Determine performance
stump_accuracy = metrics.accuracy_score(y_test, y_pred_stump)
ensemble_accuracy = metrics.accuracy_score(y_test, y_pred_ensemble)

# Print message to user
print(f"The accuracy of the stump is {stump_accuracy*100:.1f} %")
print(f"The accuracy of the ensemble is {ensemble_accuracy*100:.1f} %")


#feature_array conatins all the 420 features and label_arr contains all the labels corresponding
#   1.0264    97.99
# c and gamma are predicted using gridsearchCV

clf=svm.SVC(kernel='rbf',gamma=97.99,C=1.0264)
# clf = ExtraTreesClassifier(n_estimators = 1000,bootstrap = False)
clf.fit(features_array,labels_arr)

## Testing
d1=pandas.read_csv('test.csv')
X=d1.iloc[:,[1]].values
ID=d1.iloc[:,[0]].values
features_array=[]
for i in range(len(X)):
	seq=X[i][0]
	all_features=array_of_features_of_sequence(seq)
	features_array.append(all_features)

ans=clf.predict(features_array)



with open('output.csv', mode='w') as file:
	writer=csv.writer(file)
	writer.writerow(['ID','Label'])
	for i in range(len(ans)):
		writer.writerow([ID[i][0],ans[i]])

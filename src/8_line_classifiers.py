# from bagofwords_old import call_bow
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def get_label_dict():
	with open('./tmp/nodeid_net.txt') as fin:
		mapping = {}
		label_dict = {}
		for l in fin:
			l = l.rstrip().split(' ')
			for node in l:
				label = node.split('_')[2]
				if node not in mapping:
					id = len(mapping)
					mapping[node] = [id, label]
					label_dict[id] = label
				# print mapping[node][0], 
			# print				
	return label_dict

def load():
	mat = np.loadtxt("tmp/vec_1st_wo_norm.txt", skiprows = 1, usecols = range(1,129))
	ids = np.loadtxt("tmp/vec_1st_wo_norm.txt", skiprows = 1, usecols = (0), dtype = 'int')
	label_dict = get_label_dict()
	labels = []
	for i in ids:
		labels.append(int(label_dict[i]))
	# print mat[0], ids[0]
	# print len(labels)
	labels = np.array(labels)
	return mat, labels


def generate_metrics(predicted, actual):
	
	# print actual[:20], len(actual)
	# print predicted[:20], len(predicted)
	# return [roc_auc_score(actual, predicted), accuracy_score(actual, predicted)]
	return [accuracy_score(actual, predicted)]



def naive_bayes(train_features, train_labels, test_features, test_labels):
	gnb = GaussianNB()
	fit = gnb.fit(train_features, train_labels)
	pred = fit.predict(test_features)
	return generate_metrics(pred, test_labels)


def logistic_regression(train_features, train_labels, test_features, test_labels):
	lr = LogisticRegression()
	fit = lr.fit(train_features, train_labels)
	pred = fit.predict(test_features)
	return generate_metrics(pred, test_labels)


def svm(train_features, train_labels, test_features, test_labels):
	svm = SVC()
	fit = svm.fit(train_features, train_labels)
	pred = fit.predict(test_features)
	return generate_metrics(pred, test_labels)


def main():
	# features, labels = load()
	features, labels = load()
	# return 
	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=1)
	
	print 'AUC score with LINE vectors'
	
	#print sum([1 == test_labels[i] for i in range(len(test_labels))]) / float(len(test_labels))
	nb_res = naive_bayes(train_features, train_labels, test_features, test_labels)
	print 'Naive Bayes:\t' + str(nb_res)
	lr_res = logistic_regression(train_features, train_labels, test_features, test_labels)
	print 'Logisitic Regression:\t' + str(lr_res)
	svm_res = svm(train_features, train_labels, test_features, test_labels)
	print 'SVM:\t' + str(svm_res)


if __name__ == '__main__':
	main()
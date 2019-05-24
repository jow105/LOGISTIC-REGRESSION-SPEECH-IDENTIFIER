import sys
import math
import numpy as np
import time

def extract_label(line):
	before_tab = True
	(word,label) = ("","")
	for i in xrange(len(line)):
		if(before_tab == True):
			word += line[i]
			if(word[i] == "\t"):
				before_tab = False
		else:
			label += line[i]
	return (word,label)

def map_labels(labels_list):
	d = dict()
	i = 0
	for l in labels_list:
		if(l not in d.values() and l != ''):
			d[i] = l
			i += 1
	return d

def model1_input_train(file_name):
	input_res = np.array([])
	words = np.array([''])
	labels_list = np.array([])
	for line in open(file_name):
		line = line.strip("\n\r")
		if line == '':
			loc = np.array([0])
			input_res = np.append(input_res, {'':loc})
			labels_list = np.append(labels_list,'')
			continue
		(word,label) = extract_label(line)
		word = word.strip("\t")

		if(word not in words):
			words = np.append(words,word)

		labels_list = np.append(labels_list,label)

		input_res = np.append(input_res,{word:1})

	for i in xrange(len(input_res)):
		key = input_res[i].keys()[0]
		where = np.where(words == input_res[i].keys()[0])[0]
		if(key == ''):
			input_res[i][key] = np.array([0])
		else:
			input_res[i][key] = np.array([0, where])

	return (input_res, words, labels_list)

def model1_input_other(file_name,words):
	input_res = np.array([])
	labels_list = np.array([])
	for line in open(file_name):
		line = line.strip("\n\r")
		if line == '':
			loc = np.array([0])
			input_res = np.append(input_res,{'':loc})
			labels_list = np.append(labels_list,'')
			continue
		(word,label) = extract_label(line)
		word = word.strip("\t")

		labels_list = np.append(labels_list,label)
		input_res = np.append(input_res,{word:1})

	for i in xrange(len(input_res)):
		key = input_res[i].keys()[0]
		where = np.where(words == input_res[i].keys()[0])[0]
		if(key == ''):
			input_res[i][key] = np.array([0])
		else:
			input_res[i][key] = np.array([0,where])

	return (input_res,labels_list)

def model2_input_train(file_name):
	input_res = np.array([])
	labels_list = np.array([])
	words = np.array([''])
	words = np.append(words,"BOS")
	words = np.append(words,"EOS")
	for line in open(file_name):
		line = line.strip("\n\r")
		if line == '':
			loc = np.array([0])
			input_res = np.append(input_res,{'':loc})
			labels_list = np.append(labels_list,'')
			continue
		(word,label) = extract_label(line)
		word = word.strip("\t")

		if((("cur:"+word) not in words) and (("prev:"+word) not in words) and (("next:"+word) not in words)):
			words = np.append(words, "cur:"+word)
			words = np.append(words, "prev:"+word)
			words = np.append(words, "next:"+word)

		labels_list = np.append(labels_list,label)
		input_res = np.append(input_res,{word:1})

	for i in xrange(len(input_res)):
		key = input_res[i].keys()[0]

		if(key == ''):
			input_res[i][key] = np.array([0])
		else:
			where = np.array([0])
			if((input_res[i-1].keys()[0] == '' and input_res[i+1].keys()[0] == '') or (i == 0 and input_res[i+1].keys()[0] == '') or (input_res[i-1].keys()[0] == '' and (i == len(input_res)-1))):
				where = np.append(where,np.where(words == 'BOS'))
				where = np.append(where,np.where(words == ('cur:'+input_res[i].keys()[0])))
				where = np.append(where,np.where(words == 'EOS'))
			elif(i == 0 or input_res[i-1].keys()[0] == ''):
				where = np.append(where,np.where(words == 'BOS'))
				where = np.append(where,np.where(words == ('cur:'+input_res[i].keys()[0])))
				where = np.append(where,np.where(words == ('next:'+input_res[i+1].keys()[0])))
			elif(i == len(input_res)-1 or input_res[i+1].keys()[0] == ''):
				where = np.append(where,np.where(words == ('prev:'+input_res[i-1].keys()[0])))
				where = np.append(where,np.where(words == ('cur:'+input_res[i].keys()[0])))
				where = np.append(where,np.where(words == 'EOS'))
			else:
				where = np.append(where,np.where(words == ('prev:'+input_res[i-1].keys()[0])))
				where = np.append(where,np.where(words == ('cur:'+input_res[i].keys()[0])))
				where = np.append(where,np.where(words == ('next:'+input_res[i+1].keys()[0])))
			input_res[i][key] = where

	return (input_res,words,labels_list)

def model2_input_other(file_name, words):
	input_res = np.array([])
	labels_list = np.array([])
	for line in open(file_name):
		line = line.strip("\n\r")
		if line == '':
			loc = np.array([0])
			input_res = np.append(input_res,{'':loc})
			labels_list = np.append(labels_list,'')
			continue
		(word,label) = extract_label(line)
		word = word.strip("\t")

		labels_list = np.append(labels_list,label)
		input_res = np.append(input_res,{word:1})

	for i in xrange(len(input_res)):
		key = input_res[i].keys()[0]

		if(key == ''):
			input_res[i][key] = np.array([0])
		else:
			where = np.array([0])
			if((input_res[i-1].keys()[0] == '' and input_res[i+1].keys()[0] == '') or (i == 0 and input_res[i+1].keys()[0] == '') or (input_res[i-1].keys()[0] == '' and (i == len(input_res)-1))):
				where = np.append(where,np.where(words == 'BOS'))
				where = np.append(where,np.where(words == ('cur:'+input_res[i].keys()[0])))
				where = np.append(where,np.where(words == 'EOS'))
			elif(i == 0 or input_res[i-1].keys()[0] == ''):
				where = np.append(where,np.where(words == 'BOS'))
				where = np.append(where,np.where(words == ('cur:'+input_res[i].keys()[0])))
				where = np.append(where,np.where(words == ('next:'+input_res[i+1].keys()[0])))
			elif(i == len(input_res)-1 or input_res[i+1].keys()[0] == ''):
				where = np.append(where,np.where(words == ('prev:'+input_res[i-1].keys()[0])))
				where = np.append(where,np.where(words == ('cur:'+input_res[i].keys()[0])))
				where = np.append(where,np.where(words == 'EOS'))
			else:
				where = np.append(where,np.where(words == ('prev:'+input_res[i-1].keys()[0])))
				where = np.append(where,np.where(words == ('cur:'+input_res[i].keys()[0])))
				where = np.append(where,np.where(words == ('next:'+input_res[i+1].keys()[0])))
			input_res[i][key] = where
	return (input_res,labels_list)

# TODO: UPDATE 
def dot_prod(one_locations, theta, k):
	if(len(one_locations) == 2):
		return theta[k][one_locations[0]] + theta[k][one_locations[1]]
	elif(len(one_locations) == 1):
		return theta[k][one_locations[0]]
	else:
		tot = 0
		for loc in xrange(len(one_locations)):
			tot += theta[k][one_locations[loc]]
		return tot

def likelihood(theta, i, k, labels_dict, input_res):  # theta, i, y, x
	dict_pos = input_res[i].keys()[0]
	numerator = math.exp(dot_prod(input_res[i][dict_pos],theta,k))
	# denominator = 0
	# d = input_res[i].keys()[0]
	# for j in xrange(len(labels_dict)):
		
	# 	denominator += math.exp(dot_prod(input_res[i][d],theta,j))

	return numerator #/denominator

# TODO: UPDATE PARAMS
# N = number of samples, K = number of lables
def neg_log_likelihood(theta, labels_dict, labels_list, input_res):
	total = 0
	num = 0
	for i in xrange(len(input_res)):
		if(input_res[i].keys()[0] != ''):
			num += 1
			inner = 0
			d = input_res[i].keys()[0]
			numerator = np.array([0.0 for _ in xrange(len(labels_dict))])
			for k in xrange(len(labels_dict)):
				numerator[k] = math.exp(dot_prod(input_res[i][d],theta,k))
			denominator = np.sum(numerator)
			for k in xrange(len(labels_dict)):
				indicator = (1 if labels_dict.values().index(labels_list[i]) == k else 0)
				inner += indicator*math.log(numerator[k]/denominator)
			total += inner
	return (-1.0/num)*total



	# 	for k in xrange(len(labels_dict)):
	# 		indicator = (1 if labels_dict.values().index(labels_list[i]) == k else 0)
	# 		denominator = 0
	# 		for j in xrange(len(labels_dict)):
	# 			dict_pos = input_res[i].keys()[0]
	# 			denominator += math.exp(dot_prod(input_res[i][dict_pos],theta,j))

	# 		d = input_res[i].keys()[0]
	# 		inner += indicator*math.log(math.exp(dot_prod(input_res[i][d],theta,k))/denominator)
	# 	total += inner
	# return (-1.0/len(input_res))*total

# TODO: potential for bugs with that return statement
def gradient_update(theta, i, k, labels_dict, labels_list, input_res):
	indicator = (1 if labels_dict.values().index(labels_list[i]) == k else 0)
	d = input_res[i].keys()[0]
	numerator = math.exp(dot_prod(input_res[i][d],theta,k))
	# denominator = 0
	# for j in xrange(len(labels_dict)):
	# 	denominator += math.exp(dot_prod(input_res[i][d],theta,j))
	#d = input_res[i].keys()[0] # [first_pos, second_pos]
	# return -1*(indicator - (numerator/denominator))#*(input_res[i][d])
	return numerator

# TODO: potential for bugs with value returned from gradient_update
def SGD(theta, learning_rate, labels_dict, labels_list, input_res):
	new_theta = theta
	row_size = len(theta[0])
	updates = np.array([0.0 for _ in xrange(len(labels_dict))])
	numerator = np.array([0.0 for _ in xrange(len(labels_dict))])

	for i in xrange(len(input_res)):
		
		# print "i = " + str(i)
		# k should correspond to the index of the label of the input_res[i]
		if(input_res[i].keys()[0] != ''):
			for k in xrange(len(labels_dict)):
				numerator[k] = gradient_update(theta, i, k, labels_dict, labels_list, input_res)

			denominator = np.sum(numerator)
			# want index of labels_list[i] in labels_dict
			# updates = []
			for j in xrange(len(labels_dict)):
				indicator = (1 if labels_dict.values().index(labels_list[i]) == j else 0)
				updates[j] = -1*(indicator-(numerator[j]/denominator))
				# updates.append(gradient_update(theta, i, j, labels_dict, labels_list, input_res))

			d = input_res[i].get(input_res[i].keys()[0]) # positions of the 1 in one-hot for training data i
			# CAN PROBABLY OPTIMIZE THIS
			for k in xrange(len(labels_dict)):
				# new_row = np.array([0.0 for _ in xrange(row_size)])
				
				updated = updates[k]
				# new_row[d[0]] = updated
				# new_row[d[1]] = updated
				for idx in xrange(len(d)):
					new_theta[k][d[idx]] = new_theta[k][d[idx]] - (learning_rate*updated)
				# new_theta[k][d[0]] = new_theta[k][d[0]] - (learning_rate*updated)
				# new_theta[k][d[1]] = new_theta[k][d[1]] - (learning_rate*updated)
				# print "new_theta[" + str(k) + "] = " + str(new_theta[k])
	return new_theta

def run_epochs(num_epoch, theta, learning_rate, labels_dict, labels_list, input_res):
	new_theta = theta
	for i in xrange(num_epoch):
		print "epoch = " + str(i)
		new_theta = SGD(new_theta,learning_rate,labels_dict,labels_list,input_res)
	return new_theta

def find_max_prob(probs, labels_dict):
	best_so_far = 0
	for k in xrange(len(labels_dict)):
		if(probs[k] > probs[best_so_far]):
			best_so_far = k
		elif(probs[k] == probs[best_so_far]):
			if(labels_dict[k] > labels_dict[best_so_far]):
				best_so_far = k
	return best_so_far

def write_labels(file_name, input_res, theta, labels_dict):
	labels = np.array([])
	file = open(file_name,'w')
	for i in xrange(len(input_res)):
		if(input_res[i].keys()[0] == ''):
			labels = np.append(labels,'')
			file.write("\n")
		else:
			numerator = np.array([0.0 for _ in xrange(len(labels_dict))])
			for k in xrange(len(labels_dict)):
				numerator[k] = likelihood(theta,i,k,labels_dict,input_res)
			denominator = np.sum(numerator)

			for k in xrange(len(labels_dict)):
				numerator[k] = (numerator[k]/denominator)

			label = labels_dict[find_max_prob(numerator,labels_dict)]
			labels = np.append(labels,label)
			file.write(label + "\n")
			
	file.close()
	return labels

def write_metrics(file_name,learning_rate,num_epoch,theta,labels_dict,labels_listT,labels_listV,input_resT,input_resV):
	file = open(file_name,'w')
	new_theta = theta
	for epoch in xrange(num_epoch):
		new_theta = SGD(new_theta,learning_rate,labels_dict,labels_listT,input_resT)
		likelihoodT = neg_log_likelihood(new_theta,labels_dict,labels_listT,input_resT)
		file.write("epoch=" + str(epoch+1) + " " + "likelihood(train): " + str(likelihoodT)+"\n")
		likelihoodV = neg_log_likelihood(new_theta,labels_dict,labels_listV,input_resV)
		file.write("epoch=" + str(epoch+1) + " " + "likelihood(validation): " + str(likelihoodV)+"\n")

	file.close()
	return new_theta

def write_metrics_errors(file_name,new_train_labels,labels_listT,new_test_labels,labels_listE):
	file = open(file_name,'a')
	num_train = 0
	mistakes_train = 0
	for i in xrange(len(labels_listT)):
		if(labels_listT[i] != ''):
			num_train += 1
			if(labels_listT[i] !=  new_train_labels[i]):
				mistakes_train += 1

	num_test = 0
	mistakes_test = 0
	for j in xrange(len(labels_listE)):
		if(labels_listE[j] != ''):
			num_test += 1
			if(labels_listE[j] != new_test_labels[j]):
				mistakes_test += 1

	file.write("error(train): " + str(1.0*mistakes_train/num_train) + "\n")
	file.write("error(test): " + str(1.0*mistakes_test/num_test) + "\n")
	file.close()

if __name__ == "__main__":
	start = time.time()
	train_input = sys.argv[1]
	validation_input = sys.argv[2]
	test_input = sys.argv[3]
	train_out = sys.argv[4]
	test_out = sys.argv[5]
	metrics_out = sys.argv[6] 
	num_epoch = int(sys.argv[7])
	feature_flag = int(sys.argv[8])

	if(feature_flag == 1):
		(input_resT, words, labels_listT) = model1_input_train(train_input)
		# print input_resT
		# print "words = " + str(words)
		labels_dict = map_labels(labels_listT)
		# (input_resV, _ , labels_listV) = model1_input(validation_input)
		# (input_resE, _ , labels_listE) = model1_input(test_input)
		(input_resV, labels_listV) = model1_input_other(validation_input,words)
		(input_resE, labels_listE) = model1_input_other(test_input,words)
		theta = np.array([([1.0]+[0.0 for _ in xrange(len(words)-1)]) for _ in xrange(len(labels_dict))])
		# start_x = time.time()
		# final_theta = run_epochs(num_epoch, theta, 0.5, labels_dict, labels_listT, input_resT)
		# end_x = time.time()
		# print "RUNNING EPOCHS RUN TIME = " + str(end_x-start_x) + "s"
		
		final_theta = write_metrics(metrics_out,0.5,num_epoch,theta,labels_dict,labels_listT,labels_listV,input_resT,input_resV)

		new_train_labels = write_labels(train_out, input_resT, final_theta, labels_dict)
		new_test_labels = write_labels(test_out, input_resE, final_theta, labels_dict)

		# write_metrics(metrics_out,0.5,num_epoch,theta,labels_dict,labels_listT,labels_listV,input_resT,input_resV)
		write_metrics_errors(metrics_out,new_train_labels,labels_listT,new_test_labels,labels_listE)
		end = time.time()
		# print "total run time = " + str(end-start) + " s"

	if(feature_flag == 2):
		(input_resT, words, labels_listT) = model2_input_train(train_input)
		# print words
		# print "words = " + str(words)
		labels_dict = map_labels(labels_listT)
		# (input_resV, _ , labels_listV) = model1_input(validation_input)
		# (input_resE, _ , labels_listE) = model1_input(test_input)
		(input_resV, labels_listV) = model2_input_other(validation_input,words)
		(input_resE, labels_listE) = model2_input_other(test_input,words)
		theta = np.array([([1.0]+[0.0 for _ in xrange(len(words)-1)]) for _ in xrange(len(labels_dict))])
		# start_x = time.time()
		# final_theta = run_epochs(num_epoch, theta, 0.5, labels_dict, labels_listT, input_resT)
		# end_x = time.time()
		# print "RUNNING EPOCHS RUN TIME = " + str(end_x-start_x) + "s"
		
		final_theta = write_metrics(metrics_out,0.5,num_epoch,theta,labels_dict,labels_listT,labels_listV,input_resT,input_resV)

		new_train_labels = write_labels(train_out, input_resT, final_theta, labels_dict)
		new_test_labels = write_labels(test_out, input_resE, final_theta, labels_dict)

		# write_metrics(metrics_out,0.5,num_epoch,theta,labels_dict,labels_listT,labels_listV,input_resT,input_resV)
		write_metrics_errors(metrics_out,new_train_labels,labels_listT,new_test_labels,labels_listE)
		end = time.time()
		# print "total run time = " + str(end-start) + " s"		

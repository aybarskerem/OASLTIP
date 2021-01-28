import numpy as np
import sys
from math import sqrt

if len(sys.argv) != 3:
	print "run as: python " + sys.argv[0] + " <ground_truth file> <result_file (mu/means)>"
	exit(0)
	
gt_list=[]
with open(sys.argv[1],"r") as ground_truth:
	for line in ground_truth:
		gt_list.append( line.strip().split(",") )

result_list=[]
with open(sys.argv[2],"r") as result:
	for line in result:
		if line != "\n":
			result_list.append( line.strip().split(","))
		else:
			result_list.append(None)

errors=[]
for i in range(len(gt_list)):
	if(result_list[i] is not None):
		print gt_list[i]
		print result_list[i]
		errors.append( sqrt( (float(gt_list[i][0]) - float(result_list[i][0]) )**2 + (float(gt_list[i][1]) - float(result_list[i][1]) )**2 ) )
		

errors=np.array(errors)
number_of_errors=errors.size
print "avg accuracy is: ", np.mean(errors)
print "std is: ", np.std(errors)


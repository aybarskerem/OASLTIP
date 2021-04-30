import numpy as np
import sys
from math import sqrt

if len(sys.argv) != 3:
	print "run as: python " + sys.argv[0] + " <ground_truth file> <result_file>"
	exit(0)
	
gt_list=[]
with open(sys.argv[1],"r") as ground_truth:
	for i in range(8): # start from 8th sec (go up until end which is 57th sec)
		ground_truth.next()
        
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

with open("../../means_avgAcc.txt","a+") as means_avgAcc_file:
	means_avgAcc_file.write(str(np.mean(errors))+"\n")
with open("../../stds_avgAcc.txt","a+") as stds_avgAcc_file:
	stds_avgAcc_file.write(str(np.std(errors))+"\n")

print "avg accuracy is: ", np.mean(errors)
print "std is: ", np.std(errors)
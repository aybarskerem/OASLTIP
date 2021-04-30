import numpy as np
import sys
from math import sqrt
import os

if len(sys.argv) != 2:
	print "run as: python " + sys.argv[0] + " <ground_truth file>"
	exit(0)
	
gt_list=[]
with open(sys.argv[1],"r") as ground_truth:
	for i in range(8): # start from 8th sec (go up until end which is 57th sec)
		ground_truth.next()

	for line in ground_truth:
		gt_list.append( line.strip().split(",") )

print "gt_list is: ", gt_list

distances=[]
for i in range(len(gt_list)-1):
	distances.append( sqrt( (float(gt_list[i+1][0]) - float(gt_list[i][0]) )**2 + (float(gt_list[i+1][1]) - float(gt_list[i][1]) )**2 ) )
		
print "distances is: ", distances

distances=np.array(distances)
with open( "speed_" + os.path.splitext(os.path.basename(sys.argv[1]) )[0] + ".txt","a+") as outFile:
	outFile.write("mean speed is: " + str(np.mean(distances) ) + "\n")
	outFile.write("std of speed is: " +  str(np.std(distances)) + "\n")

print "avg accuracy is: ", np.mean(distances)
print "std is: ", np.std(distances)
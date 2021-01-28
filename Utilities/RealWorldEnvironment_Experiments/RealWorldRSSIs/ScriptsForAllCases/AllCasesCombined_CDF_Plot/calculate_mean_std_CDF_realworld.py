from __future__ import division #for floating point, integer divison operators (/ and //) 
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import sys


if len(sys.argv) != 7:
	print "run as: python " + sys.argv[0] + " <ground_truth file1> <result_file1> <ground_truth file2> <result_file2> <ground_truth file3> <result_file3>"
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

all_distances=[[],[],[]]
for i in range(len(gt_list)):
	if(result_list[i] is not None):
		#print gt_list[i]
		#print result_list[i]
		all_distances[0].append( sqrt( (float(gt_list[i][0]) - float(result_list[i][0]) )**2 + (float(gt_list[i][1]) - float(result_list[i][1]) )**2 ) )
		

gt_list=[]
with open(sys.argv[3],"r") as ground_truth:
	for line in ground_truth:
		gt_list.append( line.strip().split(",") )

result_list=[]
with open(sys.argv[4],"r") as result:
	for line in result:
		if line != "\n":
			result_list.append( line.strip().split(","))
		else:
			result_list.append(None)


for i in range(len(gt_list)):
	if(result_list[i] is not None):
		#print gt_list[i]
		#print result_list[i]
		all_distances[1].append( sqrt( (float(gt_list[i][0]) - float(result_list[i][0]) )**2 + (float(gt_list[i][1]) - float(result_list[i][1]) )**2 ) )


gt_list=[]
with open(sys.argv[5],"r") as ground_truth:
	for line in ground_truth:
		gt_list.append( line.strip().split(",") )

result_list=[]
with open(sys.argv[6],"r") as result:
	for line in result:
		if line != "\n":
			result_list.append( line.strip().split(","))
		else:
			result_list.append(None)


for i in range(len(gt_list)):
	if(result_list[i] is not None):
		#print gt_list[i]
		#print result_list[i]
		all_distances[2].append( sqrt( (float(gt_list[i][0]) - float(result_list[i][0]) )**2 + (float(gt_list[i][1]) - float(result_list[i][1]) )**2 ) )



maxLen = max(map(len, all_distances))
all_distances=np.array([dist+[np.nan]*(maxLen-len(dist)) for dist in all_distances])
#all_distances=np.array(all_distances,ndmin=2)
#all_distances=np.array([np.array(dist) for dist in all_distances],ndmin=2)
#print("all_distances is: " ,all_distances)

flattened_all_distances=np.ravel(all_distances) #use flattened instead of ravel to copy instead of reference
maxDistance=np.nanmax(flattened_all_distances)
#print("flattened is: " , flattened_all_distances)
print "avg accuracy is: ", np.nanmean(flattened_all_distances) #ignore nans
print "std is: ", np.nanstd(flattened_all_distances)

# FOR CDF CALCULATION
# all_distances are 3x50 where there are 50 samples for each case
# sort for [0,:], [1,:] etc.


fig = plt.figure()
ax = fig.add_subplot(111) #1x1 grid, 1st subplot (We have only one)
ax.set_ylabel('CDF')
#ax.legend( (sorted_all_distances[0], sorted_all_distances[1], sorted_all_distances[2]), ('y', 'z', 'k') )


markers=['o','^','s']
print("all_distances size is: ",all_distances.shape[0])
for i in range(all_distances.shape[0]): # 3 cases
	currCaseDistances=all_distances[i]
	sorted_distances=np.sort(currCaseDistances[~np.isnan(currCaseDistances)])
	print(sorted_distances)
	number_of_errors=np.size(sorted_distances) 
	print("#of erros is ", number_of_errors)
	cdf=np.array(range(1,number_of_errors+1))/number_of_errors # for each element encounter, increment the probability by 1/N where N is the #of samples 
	plt.plot(sorted_distances,cdf, markersize=5, markerfacecolor='none', marker=markers[i]) # use the same cdf for all all_distances (if necessary multiply the same array by three to have 3 same arrays)

ax.set_xlabel('Error (m)')
ax.set_ylabel('Cumulative Distribution Function (CDF)')
print("max dist is: ",maxDistance)
ax.set_xticks(np.arange(0,maxDistance+0.5,0.5))
ax.set_yticks(np.arange(0,1.1,0.1))
#ax.set_xticklabels( ('30cm', '50cm', '70cm') )

#plt.tight_layout(pad=4)
plt.subplots_adjust(left=0.1,bottom=0.1,right=0.98,top=0.83,wspace=0,hspace=0) 
ax.legend(['in-pocket, crowded', 'in-hand, crowded', 'in-pocket, non-crowded'],loc="lower left", prop={'size': 10}, bbox_to_anchor=(0, 1))
plt.savefig("CDF_Errors_RealWorld.png")
#plt.show()



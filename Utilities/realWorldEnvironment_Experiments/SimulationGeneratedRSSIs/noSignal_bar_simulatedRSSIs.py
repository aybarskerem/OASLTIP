import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv)!=2:
	print "run as 'python <script name> <output file name>'"
	exit(1)
	
materialColors = {'aluminum':'silver','iron':'black', 'concrete':'gray', 'brick':'red', 'glass':'aqua'} 
N = 3
ind = np.arange(N)  # the x locations for the groups
width = 0.6       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals=[]
with open("numberOfNoSignals.txt","r") as numberOfNoSignals:
    for numberOfNoSignal in numberOfNoSignals:
        yvals.append( int(numberOfNoSignal.strip()))

#yvals = [4, 4, 4]
rects1 = ax.bar(ind, yvals, width, color='blue')


ax.set_xlabel('Real-World Experiment Simulation Cases',fontsize=20)
ax.set_ylabel('Number of No Signal Reception',fontsize=20)
ax.set_xticks(ind+width/2)
ax.set_xticklabels( ('Case1', 'Case2', 'Case3') )
ax.tick_params(axis="x", labelsize=20) 
ax.tick_params(axis="y", labelsize=20)

ind_y = np.arange(0,30,5) 
ax.set_yticks(ind_y)

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., h, int(h),
                ha='center', va='bottom',fontsize=20)

autolabel(rects1)

#plt.show()
plt.tight_layout()
plt.savefig(sys.argv[1],dpi=fig.dpi)
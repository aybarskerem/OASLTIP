import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv)!=2:
	print "run as 'python <script name> <output file name>'"
	exit(1)
	
materialColors = {'aluminum':'silver','iron':'black', 'concrete':'gray', 'brick':'red', 'glass':'aqua'} 
N = 6
ind = np.arange(N)  # the x locations for the groups
width = 0.6       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)


means=[]
stds=[]

# read mean and standard deviation for rec 1,2,3,5,7 and 9 cases from a file.
with open("means_avgAcc.txt") as meanFile:
	for line in meanFile:
		means.append( round(float(line.strip() ),2 ) )
with open("stds_avgAcc.txt") as stdFile:
	for line in stdFile:
		stds.append( round(float(line.strip() ),2 ) )


yvals=means
yerrors=stds
max_y_values=[yvals[i] + yerrors[i] for i in range(len(yvals))] 
max_y_value=max(max_y_values)
min_y_values=[yvals[i] - yerrors[i] for i in range(len(yvals))] 
min_y_value=min(min_y_values)

#yvals = [1.09, 2.15, 3.80, 4.13, 4.87, 6.06]
#rects1 = ax.bar(ind, yvals, width, color='blue')
rects1 = ax.bar(ind, height=yvals, width=width, yerr=yerrors, xerr=0,color='blue',error_kw=dict(ecolor='black', lw=3, capsize=10, capthick=1))

ax.set_xlabel('Signal Noise',fontsize=20)
ax.set_ylabel(r'Average Error $(m)$',fontsize=20)
ax.set_xticks(ind+width/2)
ax.set_yticks(np.arange(8))
ax.set_xticklabels( ('0dBm', '5dBm', '10dBm', '15dBm', '20dBm', '30dBm') )
ax.tick_params(axis="x", labelsize=20) 
ax.tick_params(axis="y", labelsize=20)

ind_y=np.arange(min(0, min_y_value // 1),max_y_value // 1+2,1)
ax.set_yticks(ind_y)

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., h, "", 
                ha='center', va='bottom',fontsize=20)

autolabel(rects1)

print "means are: ",means
print "stds are: ",stds

fig.tight_layout()
plt.savefig(sys.argv[1],dpi=fig.dpi, bbox_inches="tight")

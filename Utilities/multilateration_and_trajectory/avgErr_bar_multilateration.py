import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv)!=2:
	print "run as 'python <script name> <output file name>'"
	exit(1)
	
N = 8
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


#yvals = [0.44, 1.17, 1.83, 0, 1.42, 2.26, 1.44]
#rects1 = ax.bar(ind[1:], yvals, width, color='blue')
#rects2= ax.bar(0, 0.01, width, color='blue')

rects1 = ax.bar(ind, height=yvals, width=width, yerr=yerrors, xerr=0,color='blue',error_kw=dict(ecolor='black', lw=3, capsize=10, capthick=1))


ax.set_xlabel('Case Number',fontsize=20)
ax.set_ylabel(r'Average Error $(m)$',fontsize=20)
ax.set_xticks(ind+width/2)
ax.set_xticklabels( ('1', '2', '3', '4', '5', '6', '7', '8') )
ax.tick_params(axis="x", labelsize=20) 
ax.tick_params(axis="y", labelsize=20)

#ax.set_yticks(np.arange(4))
ind_y=np.arange(min(0, min_y_value // 1),max_y_value // 1+2,1)
ax.set_yticks(ind_y)

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., h, "", 
                ha='center', va='bottom',fontsize=20)

def autolabel2(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/1.4, h, "", 
                ha='center', va='bottom',fontsize=20)

#autolabel2(rects2)
#autolabel(rects1)

#plt.show()
plt.savefig(sys.argv[1],dpi=fig.dpi, bbox_inches="tight")

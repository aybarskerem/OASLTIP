import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv)!=2:
	print "run as 'python <script name> <output file name>'"
	exit(1)

materialColors = {'aluminum':'silver','iron':'black', 'concrete':'gray', 'brick':'red', 'glass':'aqua'} 
N = 2
ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

stds=[]
means=[]
# read mean and standard deviation for rec 1,2,3,5,7 and 9 cases from a file.
with open("means_noSignalAcc.txt") as meanFile:
	for line in meanFile:
		means.append( round(float(line.strip() ) ) )

with open("stds_noSignalAcc.txt") as stdFile:
	for line in stdFile:
		stds.append( round(float(line.strip() ) ) )


yvals=means
yerrors=stds
max_y_values=[yvals[i] + yerrors[i] for i in range(len(yvals))] 
max_y_value=max(max_y_values)
min_y_values=[yvals[i] - yerrors[i] for i in range(len(yvals))] 
min_y_value=min(min_y_values)

print "max_y_values is: ", max_y_values
print "min_y_values is: ", min_y_values


#yvals = [0, 0]
#rects1 = ax.bar(ind, yvals, width, color='gray')
#zvals = [0, 0]
#rects2 = ax.bar(ind+width, zvals, width, color='aqua')
#kvals = [0, 0]
#rects3 = ax.bar(ind+width*2, kvals, width, color='black')

rects1 = ax.bar(ind, height=yvals[0:2], width=width, yerr=yerrors[0:2], xerr=0,color='gray', error_kw=dict(ecolor='red', lw=3, capsize=10, capthick=1))
rects2 = ax.bar(ind+width, height=yvals[2:4], width=width, yerr=yerrors[2:4], xerr=0,color='aqua', error_kw=dict(ecolor='red', lw=3, capsize=10, capthick=1))
rects3 = ax.bar(ind+width*2, height=yvals[4:6], width=width, yerr=yerrors[4:6], xerr=0,color='black', error_kw=dict(ecolor='red', lw=3, capsize=10, capthick=1))

ax.set_xlabel('Sensitivity',fontsize=20)
ax.set_ylabel('Number of No Signal Reception',fontsize=20)
ax.set_xticks(ind+width*3/2)

ax.set_xticklabels( ('0.1 meters', '0.5 meters') )
ax.tick_params(axis="x", labelsize=20) 
ax.tick_params(axis="y", labelsize=20)

plt.subplots_adjust(top=0.95)

#ax.set_yticks(np.arange(0,2) )
ind_y=np.arange(min(0, min_y_value // 1),max_y_value // 1+2,1)
ax.set_yticks(ind_y)

#ax.legend( (rects1[0], rects2[0],rects3[0]), ('Concrete', 'Glass', 'No Obstruction'),loc="lower left", prop={'size': 10}, bbox_to_anchor=(0, 1) )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., h, "", 
                ha='center', va='bottom',fontsize=20)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

print "means are: ", means
print "stds are: ",stds

#plt.show()
plt.savefig(sys.argv[1],dpi=fig.dpi, bbox_inches="tight")

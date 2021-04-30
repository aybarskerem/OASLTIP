import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv)!=2:
	print "run as 'python <script name> <output file name>'"
	exit(1)

materialColors = {'aluminum':'silver','iron':'black', 'concrete':'gray', 'brick':'red', 'glass':'aqua'} 
N = 3
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

means=[]
stds=[]


# read mean and standard deviation for concrete_30cm, concrete50cm, concrete70cm, glas30cm, glass50cm, glass70cm respectively.
with open("means_avgAcc.txt") as meanFile:
	for line in meanFile:
		print line
		means.append( round(float(line.strip()),2 ) )
with open("stds_avgAcc.txt") as stdFile:
	for line in stdFile:
		stds.append( round(float(line.strip() ),2 ) )


yvals=means
yerrors=stds
max_y_values=[yvals[i] + yerrors[i] for i in range(len(yvals))] 
max_y_value=max(max_y_values)
min_y_values=[yvals[i] - yerrors[i] for i in range(len(yvals))] 
min_y_value=min(min_y_values)

#yvals = [1.81, 3.10, 2.42, 2.04, 0.92, 1.36]
rects1 = ax.bar(ind, height=means[0:3], width=width, yerr=yerrors[0:3], xerr=0,color='gray',error_kw=dict(ecolor='black', lw=3, capsize=10, capthick=1))
rects2 = ax.bar(ind+width, height=means[3:6], width=width, yerr=yerrors[3:6], xerr=0,color='aqua',error_kw=dict(ecolor='black', lw=3, capsize=10, capthick=1))

'''
yvals = [1.77, 0.92, 1.28]
rects1 = ax.bar(ind, yvals, width, color='gray')
zvals = [1.01, 1.01, 1.77]
rects2 = ax.bar(ind+width, zvals, width, color='aqua')
'''

print "means is: ",means
print "std is: ", stds
ax.set_xlabel('Thickness',fontsize=20)
ax.set_ylabel(r'Average Error $(m)$',fontsize=20)
ax.set_xticks(ind+width)
ax.set_xticklabels( ('30cm', '50cm', '70cm') )
ax.tick_params(axis="x", labelsize=20) 
ax.tick_params(axis="y", labelsize=20)

#ind_y = np.arange(0,2.5,0.5) 
#ax.set_yticks(ind_y)

ind_y=np.arange(min(0, min_y_value // 1),max_y_value // 1+2,1)
#ind_y = np.arange(0,4,0.5) + yerrors 
ax.set_yticks(ind_y)

#ax.legend( (rects1[0], rects2[0]), ('Concrete', 'Glass'), loc="lower left", prop={'size': 10}, bbox_to_anchor=(0, 1))

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., h, "", 
                ha='center', va='bottom',fontsize=20)

autolabel(rects1)
autolabel(rects2)


#plt.show()
plt.savefig(sys.argv[1], dpi=fig.dpi, bbox_inches="tight")
import matplotlib.pyplot as plt
import numpy as np
import sys

#degrees = [r'$0\degree$',r'$50\degree$',r'$90\degree$',r'$140\degree$',r'$180\degree$',r'$230\degree$',r'$270\degree$',r'$300\degree$']
degrees = [0, 50, 90, 140, 180, 230, 270, 320]
RSSIs180 = [-35, -25, -27, -16, -21, -30, -32,-36]
RSSIs0 = [-19, -23, -37, -33, -39, -28, -36, -32]

degree0=plt.plot(degrees, RSSIs0, color='orange',label=r'Dongle at the right ($0\degree$)',linewidth=2, markersize=10, markerfacecolor='none', marker='o' )
degree180=plt.plot(degrees, RSSIs180, color='g',label=r'Dongle at the left ($180\degree$)',linewidth=2, markersize=10, markerfacecolor='none', marker='^')

plt.xlabel('Angular Position (degree)')
plt.ylabel('RSSI (dBm)')

plt.xticks( [0,60,120,180,240,300,360] )
plt.yticks( np.arange(-45,-10, 5) )

plt.subplots_adjust(top=0.83)
plt.legend(loc="lower left", prop={'size': 12}, bbox_to_anchor=(0, 1))
#plt.legend([RSSIs180, RSSIs0], [r'Dongle at the right ($0\degree$)', r'Dongle at the left ($180\degree$)'], loc="lower left", prop={'size': 10}, bbox_to_anchor=(0, 1))
plt.savefig(sys.argv[1], bbox_inches="tight")

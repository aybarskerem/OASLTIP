from __future__ import division

import numpy as np
import math # for math.ceil
import matplotlib.pyplot as plt

from numpy.linalg import norm
from numpy.random import uniform 

from scipy.stats import multivariate_normal 
from filterpy.monte_carlo import systematic_resample, multinomial_resample , residual_resample, stratified_resample # various particle resampling methods
from scipy.optimize import minimize
from scipy.optimize import fmin_tnc
from matplotlib.patches import Ellipse, Rectangle, Circle # objects to draw on the simulation map to visualize POI, receivers, blocks etc.
import matplotlib.transforms as transforms
from matplotlib import animation
from matplotlib import collections
from numpy.random import seed
from multiprocessing import Process
from collections import deque as col_deque # for the sliding windows of RAF algorithm
import copy
import time
from sklearn.cluster import KMeans

from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon


import shapely.affinity
import matplotlib.ticker as mticker

from scipy.interpolate import griddata
from scipy.interpolate import interp2d

from matplotlib import rc
import sys
rc('text', usetex=True)

################################################# OASLTIP PARAMETERS  ################################################# 

numberOfReceivers=3 # How many receivers exists in the indoor environment
sensitivityOfResult=0.1 # How many meters should be between each search points in our multilateration algorithm (to find the best position)
maxSignalError=5 # What is the maximum signal noise in terms of dBm in the indoor environment
numberOfBlocks=2 # How many block exists in the indoor environment
#blockWidth=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) / 8
#blockLength=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) / 12 
blockWidth=0.5 # in meters (0.5 meters)
blockLength=6  # in meters (6 meters)

pastCoeff=0.2 # Between 0 and 1.

totalNumberOfPeople=1 # How many POI tracking should be simulated
MinWaitingForPerson=0  # min waiting time between each person entering the indoor environment (IE)
MaxWaitingForPerson=20 # max waiting time between each person entering the indoor environment (IE)

totalIterNo=16  # How many time steps should person of interests (POIs) spend inside the indoor environment
NumberOfParticles=300 # How many particles should be used in particle filtering
xdims=(0,14) # width  (x dimension)
ydims=(0,11) # length (y dimension)


movingLimit=1.0 # How much meters in x and y axis can a person move at one time step
minUsefulSignal=-90 # min average RSSI value that is allowed to pass our prefiltering algorithm (filterAndCheckSignal)
minSignalValue=-100 # min RSSI value catchable. In simulation, if a distance value corresponds to a RSSI value lower than this number, then
                    # this signal cannot reach to a receiver. 
   

strongSignalDistance=5 # used in simulation. One of the values used to determine the radius of the ellipse
#movingTendency=np.array([0.5,0.2])
movingTendency=np.array([0.0,0.0]) # if a person has a tendency to move in a direction at a time(1s), fill this variable as (x,y) coordinates 

prevMotionRepeatProb=0.75 # Constant representing how much our past movement should be taken into account for predicting the current motion


numberOfRooms=0 # How much room are there in the indoor environment (IE)
#roomWidth=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) / 8    #Remove the comment if room(s) should be created by the simulation
#roomLength=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) / 6   #Remove the comment if room(s) should be created by the simulation
roomWidth=[5,4]     # Comment out this line if room(s) should be created by the simulation
roomLength=[12,3]   # Comment out this line if room(s) should be created by the simulation
# roomPositions = [ [6.75,7] ] 

OOIWidth=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) /20 # beacon representing the person is drawn as circle in the map(ellipse indeed, but looks like a circle due to adjustments)
OOIHeight=OOIWidth    

particleWidth=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) /400
particleHeight=particleWidth
# these blocking material positions will be added in main functions

# make receivers in square shape
receiverWidth=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) /30
receiverLength=receiverWidth

receiverPositions=[] # to be filled in main function
blockPositions=[] # to be filled in main function
roomPositions=[] # to be filled in main function

blockMaterials=[] # to be filled in main function
roomMaterials=[] # to be filled in main function

WallRoomRatio=0.125 # Variable to determine the width of the walls inside a room. We assume walls inside a room covers almost 1/8 of the room and correct the RSSI accordingly. 
roomWallWidth=roomWidth[0] * WallRoomRatio 


materials=['concrete']
#materials = ['aluminum','iron', 'concrete', 'brick', 'glass'] # blockMaterials and roomMaterials elements are chosen from this list
materialColors = {'aluminum':'silver','iron':'black', 'concrete':'gray', 'brick':'red', 'glass':'aqua'}  # https://matplotlib.org/users/colors.html
#material_SignalDisturbance_Coefficients={'aluminum':10.0, 'iron':9.0, 'concrete':8.0, 'brick':7.0, 'glass':3.0 } # signal attenuation per 1 meter in terms of dBm
material_SignalDisturbance_Coefficients={'aluminum':20.0, 'iron':18.0, 'concrete':16.0, 'brick':14.0, 'glass':6.0 } # signal attenuation per 1 meter in terms of dBm


TX_Power=0 # TX Power of the beacon that the POIs carry
rssiAtOne=TX_Power-65 # How much RSSI values is received when a receiver device is one meter away from the beacon

# Use predefined position only if you know the trajectory of the POI (Only usable in off-line usage of the OASLTIP algorithm)
predefinedPos=np.array([ [13,10.5], [12.5,9.7], [12,9.1], [11.5,8.5], [11.2,7.5],[11.2,6.6] ,[11,5.6], [10.5,4.8], [10.2,4.5], [9.5,4.1], [8.5,4.3], [7.7,4.5], [6.8,5.3],[6.3,6.1],[6.0,6.9],[5.3,7.3] ] )

# UNCOMMENT THE FOLLOWING FOUR LINES TO USE FINGERPRINTING INFORMATION IN THE OASLTIP ALGORITHM (remove the related comments in "custom_minimize" function for fingerprinting )
fingerPrintingBeaconPositions=np.array( [ [0.25,3],  [5,   5.5   ], [11.5,   3.5   ], [12.5, 9   ] ] )
fingerPrintingSignalStrengthBeaconsToReceivers=np.array([ [ -76, -73, -86, -82    ], [ -84, -81, -67, -72   ], [ -83, -77, -85, -89   ] ]) # 4 Beacon to each of the 3 receivers
InterpolatedMapForReceivers=None
RSSIinFP={} # make it a dictionary where the key is 2d position
FP_coeff=0.2

################################################# OASLTIP PARAMETERS FINISHED  ################################################# 

# NON-PARAMETER GLOBAL CONSTANT AND VARIABLES 
safetyOffset = 10**-10 #It is used to avoid division by zero error     
OverallError=0       
numberOfNotFounds=0 # how many times none of the receivers in the indoor environment are able to catch a signal of a beacon carried by a POI.
smallestFigureSideInInch=6 


def main():

    global receiverPositions, blockPositions, roomPositions, blockMaterials, roomMaterials, roomWallWidth
    xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]

    receiverPositions=getReceiverPositionsToInstall(xdims,ydims,numberOfReceivers)

    blockPositions=getBlockPositionsToInstall(xdims=xdims,ydims=ydims,numberOfBlocks=numberOfBlocks) # install blocks without overlapping
    roomPositions=getRoomPositionsToInstall(xdims=xdims,ydims=ydims,numberOfRooms=numberOfRooms,roomBoundary=roomWallWidth/2)

    #roomPositions=[[7.5,9.5],[7,3.5]]  # Uncomment and fill this variable to manually enter the room position information
    #blockPositions=[[5,3],[9,8]]       # Uncomment and fill this variable to manually enter the room position information
  
    blockMaterials=np.random.choice(materials, numberOfBlocks) # Comment out this line to let the simulation place the blocks inside the indoor environment
    roomMaterials=np.random.choice(materials, numberOfRooms)   # Comment out this line to let the simulation place the blocks inside the indoor environment

    #interpolateFingerPrintingResult() # Comment out this line to use fingerprinting information in the OASLTIP algorithm (in custom_minimize function)

    # track each POI in a new process
    AllProcesses=[]
    for i in range(totalNumberOfPeople):
        AllProcesses.append(Process(target=processFunction,args=(i,) ) )

    for proc in AllProcesses: 
        proc.start()
        sleepAmount=np.random.uniform(low=MinWaitingForPerson,high=MaxWaitingForPerson)
        #print "sleepAmount is: " + str(sleepAmount)
        time.sleep(sleepAmount)


def processFunction(i):
    xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]
    seed(i)
    macID=generateRandomMACID() # generate a different MACID for each beacon that the POIs carry to be able to distinguish them when tracking

    if (xdims[1]-xdims[0] ) < ydims[1]-ydims[0]:
        fig=plt.figure(figsize=( smallestFigureSideInInch, (ydims[1]-ydims[0])/(xdims[1]-xdims[0]) * smallestFigureSideInInch ) )
    else:
        fig=plt.figure(figsize=( (xdims[1]-xdims[0])/(ydims[1]-ydims[0]) * smallestFigureSideInInch, smallestFigureSideInInch ) )

    fig.canvas.set_window_title(macID)
    ax=fig.add_subplot(111)
    while True:
        initialPositionOfThePerson=np.random.uniform(low=[xmin,ymin], high=[xmax,ymax], size=(2))
        #print "TMP initialPositionOfThePerson for " + str(macID) + " is: " + str(initialPositionOfThePerson) 
        isCollision=False
        for blockPosition in blockPositions:
            #if checkCircleCollision_WithRectangle(tmpBeaconPos,OOIWidth,OOIHeight,blockPosition,blockWidth,blockLength):
            if checkEllipseRectangleIntersection(initialPositionOfThePerson,OOIWidth,OOIHeight,blockPosition,blockWidth,blockLength):
                isCollision=True
                break
        if not isCollision:
            for roomIndex, roomPosition in enumerate(roomPositions):
                #if checkCircleCollision_WithRectangle(tmpBeaconPos,beaconRadius,roomPosition,roomWidth,roomLength):
                #print "room wall width is: " + str(roomWallWidth)
                # use roomWallWidth/2, since linewidth expands toward outside and inside (for roomWallWidth, expands roomWallWidth/2 towards inside and roomWallWidth/2 towards outside)
                if checkEllipseRectangleIntersection(initialPositionOfThePerson,OOIWidth,OOIHeight,roomPosition,roomWidth,roomLength,boundaryForRect=roomWallWidth/2):
                    isCollision=True
                    break   
        if not isCollision:
            break

    initialPositionOfThePerson=predefinedPos[0]
    currPerson = POI(xdims,ydims,NumberOfParticles,receiverPositions,initialPositionOfThePerson)
    ani = animation.FuncAnimation(fig, animate, fargs=[ax, macID, currPerson, NumberOfParticles,xdims,ydims,maxSignalError,movingLimit,pastCoeff,
                                                    minUsefulSignal,minSignalValue,numberOfReceivers,sensitivityOfResult,
                                                    strongSignalDistance,movingTendency],interval=1000, frames=totalIterNo, repeat=False, init_func=animate_dummy_init)

    plt.show()



def checkIfCoordinateIsInMap(coords,width,height):
    xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]
    return coords[0]-width/2 >= xmin and coords[0]+width/2 <= xmax and coords[1]-height/2 >= ymin and coords[1]+height/2 <= ymax


# UNRELATED TO THE OASLTIP ALGORITHM, USED TO DRAW THE ROOM WALL IN THE SIMULATION MAP
# Borrowed from Felix's answer in https://stackoverflow.com/questions/19394505/matplotlib-expand-the-line-with-specified-width-in-data-unit/19397279
def linewidth_from_data_units(linewidth, axis):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    
    xlength = fig.bbox_inches.width * axis.get_position().width
    xvalue_range = np.diff(axis.get_xlim())

    ylength = fig.bbox_inches.height * axis.get_position().height
    yvalue_range = np.diff(axis.get_ylim())

    xlength *= 72
    ylength *= 72

    # Scale linewidth to value range
    xresult=linewidth * (xlength / xvalue_range)
    yresult=linewidth * (ylength / yvalue_range)
    return max(xresult,yresult)

# Class for the person of interest (POI). POI is the person that we track.
class POI:
    def __init__(self,xdims,ydims,NumberOfParticles,receiverPositions,initialPositionOfThePerson):

        # INITIALIZATION STEP, distribute particles on the map 
        self.particles = create_uniform_particles(xdims,ydims , NumberOfParticles) # create particles of the particle filtering algo, all around the map
        self.weights = np.ones(NumberOfParticles) / NumberOfParticles # give equal weights to all particles of particle filtering at the beginning
        self.beacon_pos=initialPositionOfThePerson
        self.prev_walkingNoise=None
        self.x_prev =  np.zeros((NumberOfParticles, 2)) # prev particles
        self.x_pp =  np.zeros((NumberOfParticles, 2)) # prev of prev particle
        self.receiverPositions = receiverPositions
        self.RSSIofReceivers=[] # RSSI value at the receiver where the signal is transmitted from the beacon that the POI carries.
        self.distToReceivers=[] # Distance of each receiver to the beacon that the POI carries. ( Distance between the POI and each of the receivers)
        self.prevCovMatrix=None # Used when drawing the confidence ellipse in the simulation.
        self.mu=None # mean of the particles. This variable will give the final most-probable position of the POI.
        self.max_weighted_particle=None # Max weighted particle of the particles used in particle filtering algorithm 
        self.slidingWindows=[col_deque([]) for i in range(len(receiverPositions) ) ] # to be filled in Running Average Filtering algorithm in function "calc_RSSIs_to_Receivers"
        
        
    # ensure person does not go out of the map and do not hit any of the obstructions in the indoor environment in the simulation
    # This function is used if we want to simulate a POI movement. Used in simulation only.
    def move_beacon_in_map(self,xdims, ydims,movingLimit,movingTendency=np.array([0,0]),roomBoundary=0 ):
        xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]

        xlow = np.maximum(xmin,self.beacon_pos[0]-movingLimit)-self.beacon_pos[0]
        xhigh =np.minimum(xmax, self.beacon_pos[0]+movingLimit)-self.beacon_pos[0]
        ylow = np.maximum(ymin,self.beacon_pos[1]-movingLimit)-self.beacon_pos[1]
        yhigh =np.minimum(ymax, self.beacon_pos[1]+movingLimit)-self.beacon_pos[1]

        while True:
            walking_noise_x = np.random.uniform(low=xlow,high=xhigh) # human motion undeterminism  
            walking_noise_y = np.random.uniform(low=ylow,high=yhigh)
            walkingNoise=np.array( (walking_noise_x,walking_noise_y)).T

            if self.prev_walkingNoise is not None:
                walkingChoices=[walkingNoise,self.prev_walkingNoise]
                walkingNoise = np.copy(walkingChoices[ np.random.choice([0,1], p=(1-prevMotionRepeatProb,prevMotionRepeatProb)) ] ) # choose the prev motion with a higher probability    
            tmpBeaconPos=self.beacon_pos + walkingNoise + movingTendency

            isCollision=not checkIfCoordinateIsInMap(tmpBeaconPos, OOIWidth,OOIHeight)
            if not isCollision:
                for blockPosition in blockPositions:
                    if checkEllipseRectangleIntersection(tmpBeaconPos,OOIWidth,OOIHeight,blockPosition,blockWidth,blockLength) or \
                    findRectangleLineSegmentIntersectionPoints(self.beacon_pos,tmpBeaconPos,blockPosition,blockWidth,blockLength) is not None :
                        isCollision=True
                        break
            if not isCollision:
                for roomIndex, roomPosition in enumerate(roomPositions):
                    #if checkCircleCollision_WithRectangle(tmpBeaconPos,beaconRadius,roomPosition,roomWidth,roomLength):
                    if checkEllipseRectangleIntersection(tmpBeaconPos,OOIWidth,OOIHeight,roomPosition,roomWidth,roomLength,boundaryForRect=roomBoundary) or \
                    indRectangleLineSegmentIntersectionPoints(self.beacon_pos,tmpBeaconPos,roomPosition,roomWidth,roomLength) is not None :
                        isCollision=True
                        break   
            if not isCollision:
                break

        self.prev_walkingNoise=np.copy(walkingNoise)
        self.beacon_pos = np.copy(tmpBeaconPos)

    # apply constant velocity model to move the POI around the map (consider the prev velocities when calculating the current motion)
    # x_prev = x(t-1)
    # x_pp = prev of x_prev
    def predict_BLE( self, no_of_noise_elements, movingLimit, pastCoeff, xdims, ydims, movingTendency=np.array([0,0]) ):
        xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]
        # ALL PARTICLES SHOULD RESIDE IN THE MAP, CHECK FOR BEING INSIDE FOR EACH PARTICLE (MOVE THAT AMOUNT AT THE BORDERS AT MAX)
        # min of x, should not be lower than map's xmin && max of x should not be larger than map's xmax
        xlow = np.maximum(xmin,self.particles[:,0]-movingLimit)-self.particles[:,0]
        xhigh =np.minimum(xmax, self.particles[:,0]+movingLimit)-self.particles[:,0]
        ylow = np.maximum(ymin,self.particles[:,1]-movingLimit)-self.particles[:,1]
        yhigh =np.minimum(ymax, self.particles[:,1]+movingLimit)-self.particles[:,1]
        walking_noise_x = np.random.uniform(low=xlow,high=xhigh,size=self.particles.shape[0]) # human motion undeterminism  
        walking_noise_y = np.random.uniform(low=ylow,high=yhigh,size=self.particles.shape[0])
        walking_noise_x=np.array(walking_noise_x)
        walking_noise_y=np.array(walking_noise_y)
        walking_noise=np.array( (walking_noise_x,walking_noise_y)).T 
       
        if np.count_nonzero(self.x_prev) != 0 and np.count_nonzero(self.x_pp) != 0:
            past_velocity = self.x_prev - self.x_pp  # Past Position - Past of past position. It gives the velocity of the POI at the previous time step
            change_in_pos = (1-pastCoeff) * walking_noise + pastCoeff * past_velocity  # constant_velocity_motion
        else:
            change_in_pos = walking_noise
        self.particles += change_in_pos + movingTendency    


    # Update the weight of the particles according to the measured beacon position found in the multilateration algorithm for the current time step
    def update_weights(self):
        distances = np.linalg.norm(self.particles - self.averaged_beacon_pos, axis=1)

        self.weights *= np.sum(distances)/distances
        self.weights += 10**(-30)      # avoid round-off to zero
        self.weights /= sum(self.weights) # normalize

    # Resample N_eff
    def resample_from_higher_weights(self,tmp_particles, tmp_weights):
        #indices = multinomial_resample(weights)
        #indices = residual_resample(weights)
        #indices = stratified_resample(weights)
        indices = systematic_resample(self.weights) # Use any of the resampling methods above.

        tmp_particles[:] = tmp_particles[indices]
        tmp_weights[:] = tmp_weights[indices]
        tmp_weights.fill(1.0 / len(tmp_weights))

    # PREFIILTER FUNCTION FOR  THE INCOMING SIGNAL ( and if in simulation, simulate the weaking of the signal before even arriving at the receivers)
    def calc_RSSIs_to_Receivers(self,minSignalValue,minUsefulSignal,maxSignalError):

        receiverIndex=0
        self.RSSIofReceivers[:] = [] # empty all RSSIs to accept the new ones for the current time step
        
        for receiverPosition in self.receiverPositions:
            RSSI = 0
            if(maxSignalError > 0):
                RSSI=weakenedSignal( distance_to_RSSI( np.linalg.norm(receiverPosition-self.beacon_pos) ) , maxSignalError ) 
            else:
                RSSI=distance_to_RSSI( np.linalg.norm(receiverPosition-self.beacon_pos ) )

            isCollision=False
         

            # Comment out "SIGNAL WEAKING CODE BLOCK BELOW TO USE OASLTIP FOR A REAL-WORLD APPLICATION SINCE IN REAL-WORLD, WEAKEANING NATURALLY HAPPENS"
            ##########################  WEAKENING THE SIGNAL DUE TO THE SIGNAL HITTING AN OBSTRUCTION (BLOCK OR ROOM) ##########################
            # this is used to weaken the signal in case there was a block or room between the receiver and the beacon
            # this simulates the signal before we catch it in real life.            
            weakeningAmount=0 # distance between the receiver and the beacon / 1 meter *  ( how many dBm to reduce for 1 meter)
            for blockIndex, blockPosition in enumerate(blockPositions):
                receiverBeaconBlockIntersection=findRectangleLineSegmentIntersectionPoints(receiverPosition,self.beacon_pos,blockPosition,blockWidth,blockLength)
                
                if receiverBeaconBlockIntersection is not None:
                    isCollision=True
                    weakeningAmount+=np.linalg.norm(receiverBeaconBlockIntersection[0,:]-receiverBeaconBlockIntersection[1,:]) * material_SignalDisturbance_Coefficients[ blockMaterials[blockIndex] ] * np.random.uniform(0.5,1.5)  # +- some noise
            
            for roomIndex, roomPosition in enumerate(roomPositions):
                receiverBeaconRoomIntersection=findRectangleLineSegmentIntersectionPoints(receiverPosition,self.beacon_pos,roomPosition,roomWidth,roomLength)
                if receiverBeaconRoomIntersection is not None:
                    isCollision=True
                    weakeningAmount+=np.linalg.norm(receiverBeaconRoomIntersection[0,:]-receiverBeaconRoomIntersection[1,:]) * WallRoomRatio * material_SignalDisturbance_Coefficients[ roomMaterials[roomIndex] ] * np.random.uniform(0.5,1.5)
            ###################################################### SIGNAL WEAKING FINISHED ######################################################


            if isCollision:
                #print "No Line Of Sight between receiver " + str(receiverPosition) + " and beacon " + str(self.beacon_pos)
                RSSI-=weakeningAmount
            else:   
                #print "Direct Line Of Sight between receiver " + str(receiverPosition) + " and beacon " + str(self.beacon_pos) 
                pass
                

            ########################################## RUNNING AVERAGE FILTERING (RAF) ALGORITHM STARTS ##########################################
            

            # each receiver should have a sliding window
            # max slidingWindows size should be 7
            slidingWindow = self.slidingWindows[receiverIndex]
            while len(slidingWindow) >=7:
                slidingWindow.popleft() # delete oldest element
            slidingWindow.append(RSSI) # appends at the right

            if self.filterAndCheckSignal(minUsefulSignal,receiverIndex) and RSSI > minSignalValue:
                #print "filtering was successful"
                self.RSSIofReceivers.append( RSSI )
            else:
                #print "filtering was not successful"
                self.RSSIofReceivers.append( None )

            receiverIndex+=1


    def filterAndCheckSignal(self,minUsefulSignal,receiverIndex):
        mean=0.0
        sum=0.0
        slidingWindow = self.slidingWindows[receiverIndex]
        if len(slidingWindow) < 3:
            return False
        else:
            noOutlierDeque=col_deque(sorted(slidingWindow) )
            noOutlierDeque.popleft() # delete smallest
            noOutlierDeque.pop() # delete greatest

            for signalVal in noOutlierDeque:
                sum+=signalVal
            mean=sum/len(noOutlierDeque)

        return mean >= minUsefulSignal


    # if RSSI is lower than minUsefulSignal , then omit this receiver ( assuming we use 0dBm signal powered beacons)
    def setBeaconDistances_fromRSSIs(self,minUsefulSignal):
        self.distToReceivers[:] = []
        for RSSIofReceiver in self.RSSIofReceivers:
            if RSSIofReceiver is not None and  \
            RSSIofReceiver > minUsefulSignal:
                self.distToReceivers.append( RSSI_to_distance( RSSIofReceiver ) + safetyOffset ) # add safetyOffset0 to avoid divide by zero in the custom_minimize function
            else:
                self.distToReceivers.append( None )

    def multiLateration(self,xdims,ydims,sensitivityOfResult):

        receiverPositionsArray=np.array(self.receiverPositions)
        resultingPoint = custom_minimize(self.RSSIofReceivers,np.vstack(receiverPositionsArray ),xdims,ydims,sensitivityOfResult,checkForBlocks=True )
        return resultingPoint


    def calc_PDF(self,strongSignalDistance,pastCoeff):

        numberOfNotNones=0
        numberOfStrongSignals=0
        confidenceEllipseMultiplier=1

        for distToReceiver in self.distToReceivers:
            if distToReceiver is not None:
                numberOfNotNones+=1
                #print "dist to receiver is: " + str(distToReceiver)
                if distToReceiver < strongSignalDistance:
                    numberOfStrongSignals+=1


        # Calculate mean of the particles, the covariance matrix to determine the confidence ellipse size and max weighted particle.
        self.mu = np.average(self.particles, weights=self.weights, axis=0)
        self.covMatrix = np.cov(m=self.particles, rowvar=False, aweights=self.weights)  # rowvar has to be False otherwise each row represents a variable, with observations in the columns. 
                                                                                        # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.cov.html
        self.max_weighted_particle =  self.particles[np.argmax(self.weights) ]

        # PUNISH (ENLARGEN THE CONFIDENCE ELLIPSE) if we have little amount of RSSI data or if the BLE signals are not strong enough
        if numberOfNotNones >=3:
            if numberOfStrongSignals >= 3:
                confidenceEllipseMultiplier=1 # No change
            elif numberOfStrongSignals == 2:
                confidenceEllipseMultiplier=1.25
            elif numberOfStrongSignals == 1:
                confidenceEllipseMultiplier=1.5
            else: # numberOfStrongSignals == 0
                confidenceEllipseMultiplier=2

        # x1.6 worse than the >=3 case
        elif numberOfNotNones == 2:
            if numberOfStrongSignals == 2:
                confidenceEllipseMultiplier=2
            elif numberOfStrongSignals == 1:
                confidenceEllipseMultiplier=2.4
            else: # numberOfStrongSignals == 0
                confidenceEllipseMultiplier=3.2

        # x3 worse than the >=3 case
        elif numberOfNotNones == 1:
            if numberOfStrongSignals == 1:
                confidenceEllipseMultiplier=4.5
            else: # numberOfStrongSignals == 0
                confidenceEllipseMultiplier=6.0

        # x5 worse than the >=3 case
        else: # numberOfNotNones == 0:
            #confidenceEllipseMultiplier=float("inf") # do not draw the ellipse at all
            confidenceEllipseMultiplier=10.0 # 10 is a high number, we may not see the ellipse in the indoor environment due to our low confidence about the POI position

        self.covMatrix*=confidenceEllipseMultiplier

        # if pastCoeff == 1, then except for the first time step, covMatrix will be the same as prev cov matrix 
        if self.prevCovMatrix is not None:
            self.covMatrix=self.covMatrix*(1-pastCoeff) + pastCoeff*self.prevCovMatrix
    

# circle center, circle radius, 2 ends of line segment
def findEllipseLineSegmentIntersectionPoints(ellipseCenter,width,height, p1,p2):

    if ( np.array_equal(p1,p2) ):
        return None

    centerPoint = Point(ellipseCenter)
    unitCircle = centerPoint.buffer(1).boundary
    ellipse=shapely.affinity.scale(unitCircle,width,height)
    line = LineString([p1,p2])

    if ellipse.intersects(line):
        intersectionPointObject = ellipse.intersection(line)
        intersectionPoint=np.array([intersectionPointObject.coords[0],intersectionPointObject.coords[1]])
    else:
        intersectionPoint=None

    return intersectionPoint


def checkFirstRectangleContainsSecondRectangle(rectCenter,rectWidth,rectLength, rectCenter2,rectWidth2,rectLength2,boundaryForFirstRect=0,boundaryForSecondRect=0):
    bottomLeftCorner=rectCenter+np.array([-(rectWidth/2 + boundaryForFirstRect),-(rectLength/2 + boundaryForFirstRect) ])
    topLeftCorner=rectCenter+np.array([-(rectWidth/2 + boundaryForFirstRect) ,rectLength/2 + boundaryForFirstRect])
    bottomRightCorner=rectCenter+np.array([rectWidth/2 + boundaryForFirstRect,-(rectLength/2 + boundaryForFirstRect) ])
    topRightCorner=rectCenter+np.array([rectWidth/2 + boundaryForFirstRect,rectLength/2 + boundaryForFirstRect])

    bottomLeftCorner2=rectCenter2+np.array([-(rectWidth2/2 + boundaryForSecondRect),-(rectLength2/2 + boundaryForSecondRect) ])
    topLeftCorner2=rectCenter2+np.array([-(rectWidth2/2 + boundaryForSecondRect) ,rectLength2/2 + boundaryForSecondRect])
    bottomRightCorner2=rectCenter2+np.array([rectWidth2/2 + boundaryForSecondRect,-(rectLength2/2 + boundaryForSecondRect) ])
    topRightCorner2=rectCenter2+np.array([rectWidth2/2 + boundaryForSecondRect,rectLength2/2 + boundaryForSecondRect])

    rectangle = Polygon([bottomLeftCorner, topLeftCorner, topRightCorner, bottomRightCorner])
    rectangle2 = Polygon([bottomLeftCorner2, topLeftCorner2, topRightCorner2, bottomRightCorner2])

    return rectangle.contains(rectangle2)

def checkRectangleRectangleIntersection(rectCenter,rectWidth,rectLength, rectCenter2,rectWidth2,rectLength2,boundaryForFirstRect=0,boundaryForSecondRect=0):
    bottomLeftCorner=rectCenter+np.array([-(rectWidth/2 + boundaryForFirstRect),-(rectLength/2 + boundaryForFirstRect) ])
    topLeftCorner=rectCenter+np.array([-(rectWidth/2 + boundaryForFirstRect) ,rectLength/2 + boundaryForFirstRect])
    bottomRightCorner=rectCenter+np.array([rectWidth/2 + boundaryForFirstRect,-(rectLength/2 + boundaryForFirstRect) ])
    topRightCorner=rectCenter+np.array([rectWidth/2 + boundaryForFirstRect,rectLength/2 + boundaryForFirstRect])

    bottomLeftCorner2=rectCenter2+np.array([-(rectWidth2/2 + boundaryForSecondRect),-(rectLength2/2 + boundaryForSecondRect) ])
    topLeftCorner2=rectCenter2+np.array([-(rectWidth2/2 + boundaryForSecondRect) ,rectLength2/2 + boundaryForSecondRect])
    bottomRightCorner2=rectCenter2+np.array([rectWidth2/2 + boundaryForSecondRect,-(rectLength2/2 + boundaryForSecondRect) ])
    topRightCorner2=rectCenter2+np.array([rectWidth2/2 + boundaryForSecondRect,rectLength2/2 + boundaryForSecondRect])

    rectangle = Polygon([bottomLeftCorner, topLeftCorner, topRightCorner, bottomRightCorner])
    rectangle2 = Polygon([bottomLeftCorner2, topLeftCorner2, topRightCorner2, bottomRightCorner2])
    return rectangle.intersects(rectangle2)

# circle center, circle radius, 2 ends of line segment
def checkEllipseRectangleIntersection(ellipseCenter,width,height, rectCenter,rectWidth,rectLength,boundaryForRect=0):

    # CORNERS
    bottomLeftCorner=rectCenter+np.array([-(rectWidth/2 + boundaryForRect),-(rectLength/2 + boundaryForRect) ])
    topLeftCorner=rectCenter+np.array([-(rectWidth/2 + boundaryForRect) ,rectLength/2 + boundaryForRect])
    bottomRightCorner=rectCenter+np.array([rectWidth/2 + boundaryForRect,-(rectLength/2 + boundaryForRect) ])
    topRightCorner=rectCenter+np.array([rectWidth/2 + boundaryForRect,rectLength/2 + boundaryForRect])

    centerPoint = Point(ellipseCenter)
    unitCircle = centerPoint.buffer(1).boundary
    ellipse=shapely.affinity.scale(unitCircle,width,height)

    rectangle = Polygon([bottomLeftCorner, topLeftCorner, topRightCorner, bottomRightCorner])
    return ellipse.intersects(rectangle)



def checkPointInsideRectangle(point,rectCenter,rectWidth,rectLength,boundaryForRect=0): 
    bottomLeftCorner=rectCenter+np.array([-(rectWidth/2 + boundaryForRect),-(rectLength/2 + boundaryForRect) ])
    topLeftCorner=rectCenter+np.array([-(rectWidth/2 + boundaryForRect) ,rectLength/2 + boundaryForRect])
    bottomRightCorner=rectCenter+np.array([rectWidth/2 + boundaryForRect,-(rectLength/2 + boundaryForRect) ])
    topRightCorner=rectCenter+np.array([rectWidth/2 + boundaryForRect,rectLength/2 + boundaryForRect])

    point = Point(point)
    rectangle = Polygon([bottomLeftCorner, topLeftCorner, topRightCorner, bottomRightCorner])
    return point.intersects(rectangle)


def findRectangleLineSegmentIntersectionPoints(p1,p2,rectCenter,rectWidth,rectLength,boundaryForRect=0):
   
    if np.array_equal(p1,p2):
        return None

    # CORNERS
    bottomLeftCorner=rectCenter+np.array([-(rectWidth/2 + boundaryForRect),-(rectLength/2 + boundaryForRect) ])
    topLeftCorner=rectCenter+np.array([-(rectWidth/2 + boundaryForRect) ,rectLength/2 + boundaryForRect])
    bottomRightCorner=rectCenter+np.array([rectWidth/2 + boundaryForRect,-(rectLength/2 + boundaryForRect) ])
    topRightCorner=rectCenter+np.array([rectWidth/2 + boundaryForRect,rectLength/2 + boundaryForRect])

    line = LineString([p1,p2])
    rectangle = Polygon([bottomLeftCorner, topLeftCorner, topRightCorner, bottomRightCorner])

    if rectangle.intersects(line):
        intersectionPointObject = rectangle.intersection(line)

        if np.array_equal(np.array(intersectionPointObject.coords).shape,np.array([2, 2])):
            intersectionPoint=np.array([intersectionPointObject.coords[0],intersectionPointObject.coords[1]])
        else:
            intersectionPoint=None
    else:
        intersectionPoint=None

    return intersectionPoint


def generateRandomMACID():
    return ':'.join('%02x'%np.random.randint(0,256) for _ in range(6))

def custom_minimize(RSSIofReceivers, receiverPositions,xdims,ydims,sensitivityOfResult=1.0,checkForBlocks=True):
    mysum=float("inf")
    maxCatchableSignalDistance = RSSI_to_distance( minUsefulSignal ) + safetyOffset
    resultingPoint=[-1,-1] # some impossible coordinate value to initialize
    for x in np.arange(xdims[0],xdims[1],sensitivityOfResult):
        for y in np.arange(ydims[0],ydims[1],sensitivityOfResult):
            # if x,y collides with a block or room, this position would not be possible 
            isPointOnObstacle=False
            for blockPosition in blockPositions:  # it will not enter this loop if there are no blocks
                if checkPointInsideRectangle([x,y],blockPosition,blockWidth,blockLength):
                    isPointOnObstacle=True
                    break
            if not isPointOnObstacle:
                for roomIndex,roomPosition in enumerate(roomPositions):
                    if checkPointInsideRectangle([x,y],roomPosition,roomWidth[roomIndex],roomLength[roomIndex]):
                        isPointOnObstacle=True
                        break
            if isPointOnObstacle:
                continue # if a point is on an obstacle, then this point cannot be what we are looking for (since the POI cannot be on an obstacle)    
            
            tmp_sum=0
            for i in range(len(receiverPositions)):  
                strengtheningAmount=0
                for blockIndex, blockPosition in enumerate(blockPositions): 
                    receiverMeanBlockIntersection = findRectangleLineSegmentIntersectionPoints(receiverPositions[i],np.array([x,y]),blockPosition,blockWidth,blockLength)
                    if receiverMeanBlockIntersection is not None:
                        strengtheningAmount+=np.linalg.norm(receiverMeanBlockIntersection[0,:]-receiverMeanBlockIntersection[1,:]) * material_SignalDisturbance_Coefficients[ blockMaterials[blockIndex] ] 
                for roomIndex, roomPosition in enumerate(roomPositions):
                    receiverMeanRoomIntersection = findRectangleLineSegmentIntersectionPoints(receiverPositions[i],np.array([x,y]),roomPosition,roomWidth[roomIndex],roomLength[roomIndex])
                    if receiverMeanRoomIntersection is not None:
                        strengtheningAmount+=np.linalg.norm(receiverMeanRoomIntersection[0,:]-receiverMeanRoomIntersection[1,:]) * WallRoomRatio * material_SignalDisturbance_Coefficients[ roomMaterials[roomIndex] ]   
                
                #xyDistToRecInFP = RSSI_to_distance(RSSIinFP[i,x,y] + strengtheningAmount ) # Remove the comment TO TAKE FINGERPRINTING DATA INTO CONSIDERATION
                xyDistToRec = np.linalg.norm( [x,y] - receiverPositions[i] )
                # Rule 1) PUNISH THE x,y points whose distances are not compatible with the RSSI values we receive.
                # Rule 2) Punish more when distToReceiverGivenRSSI is low since low distToReceiverGivenRSSI means high RSSI and high RSSIs are reliable.
                # and if x,y deviates from what low RSSIs tell us, then we should punish this x,y point.
                if RSSIofReceivers[i] is not None:
                    distToReceiverGivenRSSI=RSSI_to_distance( RSSIofReceivers[i] + strengtheningAmount) + safetyOffset
                    tmp_sum+=( abs( xyDistToRec - distToReceiverGivenRSSI ) / distToReceiverGivenRSSI ) ** 2 
                    
                    # Remove the commentS FOR THE FOLLOWING TWO LINES TO TAKE FINGERPRINTING DATA INTO CONSIDERATION
                    # if abs( RSSIofReceivers[i] - RSSIinFP[i,x,y] ) > maxSignalError: # if the difference is more than 5dBm for example:
                    #    tmp_sum+=FP_coeff*(  abs( xyDistToRecInFP - distToReceiverGivenRSSI ) / distToReceiverGivenRSSI  ) ** 2

                # Rule 1) PUNISH THE x,y points which are close to the receivers since x,y should not be close if our receivers cannot catch a signal
                else:  # If a receiver device (RSSIofReceivers[i]) is not able to catch a signal for the current time step
                    maxCatchableSignalDistance = RSSI_to_distance( minUsefulSignal + strengtheningAmount) + safetyOffset
                    if xyDistToRec < maxCatchableSignalDistance: # we see it as None, so it should not be closer than maxCatchableSignalDistance. If so, then punish
                        tmp_sum+=( abs( xyDistToRec - maxCatchableSignalDistance )  / xyDistToRec ) ** 2
                    
                    # Remove the commentS FOR THE FOLLOWING TWO LINES TO TAKE FINGERPRINTING DATA INTO CONSIDERATION
                    #if xyDistToRecInFP - maxCatchableSignalDistance: 
                    #    tmp_sum+=FP_coeff*(  abs( xyDistToRecInFP - maxCatchableSignalDistance ) / xyDistToRecInFP ) ** 2

            if tmp_sum < mysum:
                mysum = tmp_sum
                resultingPoint=[x,y]

    return resultingPoint


# after signal transmitted, maybe the signal hit a wall and reduced in strength/
# since we cannot manipulate after transmittion is node, we reduce the signal when transmitting assuming it will hit something by a posibility
# We have to increase it by a possibility
def weakenedSignal(RSSI,maxSignalError):
    return RSSI - uniform(0,maxSignalError)


def create_uniform_particles(x_range, y_range, NumberOfParticles):
    particles = np.empty((NumberOfParticles, 2)) 
    particles[:, 0] = uniform(x_range[0], x_range[1], size=NumberOfParticles)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=NumberOfParticles)
    return particles


# for each receiver hold a separate strength map
# each beacon should have its interpolation all around the map. Then we we should take weighted average of these beacons signal strength values
# For example, FOR RECEIVER 1,  if beacon1 is at [5,5] and beacon2 is at [10,3] and the point that we want to interpolate is at [10,5]. Beacon2 should have higher vote to determine signal strength
# signal strength values of the beacons (fingerprinting positions) are different for each receiver, therefore for each receiver we should hold another map info
def interpolateFingerPrintingResult():
    
    xElems=np.arange(xdims[0],xdims[1],sensitivityOfResult)
    yElems=np.arange(ydims[0],ydims[1],sensitivityOfResult  )

    allPosDistancesToReceivers={} # make it a dictionary where the key is 2d position
    for i in range(numberOfReceivers):
        for x in xElems:
            for y in yElems:
                allPosDistancesToReceivers[i,x,y]=np.linalg.norm(receiverPositions[i]- np.array([x,y]) )

    numberOfBeacons=fingerPrintingSignalStrengthBeaconsToReceivers.shape[1]
    allPosDistancesToBeacons={} # make it a dictionary where the key is 2d position
    for k in range(numberOfBeacons):
        for x in xElems:
            for y in yElems:
                allPosDistancesToBeacons[k,x,y]=np.linalg.norm(fingerPrintingBeaconPositions[k]- np.array([x,y]) )

    # INITIALIZE INTERPOLATION MAP FOR EACH RECEIVER
    global RSSIinFP
    for i in range(numberOfReceivers):
        for x in xElems:
            for y in yElems:
                minDist=np.float('inf')
                min_k=0
                # find the closest beacon to [x,y]
                # whichever beacon or receiver is the closest to [x,y], it should determine the interpolation result
                for k in range(numberOfBeacons):
                    if allPosDistancesToBeacons[k,x,y] < minDist:
                        min_k=k
                        minDist = allPosDistancesToBeacons[k,x,y]

                base_dist=np.linalg.norm(fingerPrintingBeaconPositions[min_k]-receiverPositions[i]) 
                target_dist=allPosDistancesToReceivers[i,x,y]
                base_RSSI=fingerPrintingSignalStrengthBeaconsToReceivers[i][min_k]
        
                RSSIinFP[i,x,y]=calc_relative_RSSI(base_dist,target_dist,base_RSSI)

def calc_relative_RSSI(base_dist, target_dist, base_RSSI):
    print "calc_relative_RSSI: " + str( np.log ( (target_dist+safetyOffset) / (base_dist+safetyOffset) ) ) 
    if target_dist >= 1:
        return base_RSSI + -20 * np.log ( (target_dist) / (base_dist+safetyOffset) )
    else:
        return zero_one_meter_distance_to_RSSI(target_dist)

#distance in meters, returns RSSI in dBm
# assuming signal propogation constant is 2, https://www.rn.inf.tu-dresden.de/dargie/papers/icwcuca.pdf in equation (4)
def distance_to_RSSI(distance):
    res_RSSI = 0
    if distance >=1:
        res_RSSI = -20 * np.log10(distance) + rssiAtOne
    else:
        res_RSSI = zero_one_meter_distance_to_RSSI(distance)

    return float(res_RSSI)

#RSSI in dBm, returns distance in meter
def RSSI_to_distance(RSSI):
    res_distance = 0
    if RSSI <= rssiAtOne:
        res_distance =  10**( (RSSI-rssiAtOne) / -20 )
    else:
        res_distance =  zero_one_meter_RSSI_to_distance(RSSI)

    return float(res_distance)


# EXPONENTIAL FUNCITON BETWEEN 0 and 1
def zero_one_meter_RSSI_to_distance(RSSI):
    return 10**( ( ( RSSI - TX_Power ) * np.log10(2) ) / (rssiAtOne - TX_Power) )  -1

# should return something between TX power and rssiAtOne
def zero_one_meter_distance_to_RSSI  (dist):
    return float( TX_Power + (rssiAtOne - TX_Power) * ( (np.log10(dist+1)) / (np.log10(2) ) ) )


# N_eff : Effective weight number
def neff(weights):
    return 1.0 / np.sum(np.square(weights))


def getReceiverPositionsToInstall(xdims,ydims,numberOfReceivers):
    xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]
    step_size=(1/( np.ceil(np.sqrt(numberOfReceivers*1000) ) ) )

    initial_points = np.mgrid[0:1+step_size:step_size, 0:1+step_size:step_size].reshape(2,-1).T
    receiverPositions = KMeans(n_clusters=numberOfReceivers, random_state=0,n_init=100).fit(initial_points).cluster_centers_
    if receiverPositions is not None:
        receiverPositions[:,0]=xmin+receiverPositions[:,0]*(xmax-xmin)
        receiverPositions[:,1]=ymin+receiverPositions[:,1]*(ymax-ymin)
    
    return receiverPositions


def getBlockPositionsToInstall(xdims,ydims,numberOfBlocks):
    xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]
    numberOfBlocksCreated=0
    blockPositionsToInstall=[]
    while numberOfBlocksCreated!=numberOfBlocks:
        blockCoord=np.random.uniform(low=[xmin,ymin], high=[xmax,ymax])
        collisionExists=False
        for receiverPosition in receiverPositions:
            if checkRectangleRectangleIntersection(blockCoord,blockWidth,blockLength,receiverPosition,receiverWidth,receiverLength):
                collisionExists=True
                break

        intersectionWithOtherBlocksExists=False
        if not collisionExists: # if collision exists, do not make other checks  
            for blockPosition in blockPositionsToInstall:
                if checkRectangleRectangleIntersection(blockCoord,blockWidth,blockLength,blockPosition,blockWidth,blockLength):
                    intersectionWithOtherBlocksExists=True
                    break

        if not collisionExists and not intersectionWithOtherBlocksExists:
            blockPositionsToInstall.append(blockCoord)
            numberOfBlocksCreated+=1

    return np.array(blockPositionsToInstall)


def getRoomPositionsToInstall(xdims,ydims,numberOfRooms,roomBoundary):
    xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]
    numberOfRoomsCreated=0
    roomPositionsToInstall=[]
    while numberOfRoomsCreated!=numberOfRooms:
        roomCoord=np.random.uniform(low=[xmin,ymin], high=[xmax,ymax])
        receiverHollowRoomCollisionExists=False
        for receiverPosition in receiverPositions:
            if not checkFirstRectangleContainsSecondRectangle(roomCoord,roomWidth,roomLength,receiverPosition,receiverWidth,receiverLength,boundaryForFirstRect=-roomBoundary) and \
            checkRectangleRectangleIntersection(roomCoord,roomWidth,roomLength,receiverPosition,receiverWidth,receiverLength,boundaryForFirstRect=roomBoundary):
                receiverHollowRoomCollisionExists=True
                break

        intersectionWithBlocksExists=False
        if not receiverHollowRoomCollisionExists:
            for blockPosition in blockPositions:
                if checkRectangleRectangleIntersection(roomCoord,roomWidth,roomLength,blockPosition,blockWidth,blockLength,boundaryForFirstRect=roomBoundary):
                    intersectionWithBlocksExists=True
                    break

        intersectionWithOtherRoomsExists=False
        if not receiverHollowRoomCollisionExists and not intersectionWithBlocksExists:
            for roomPosition in roomPositionsToInstall:
                if checkRectangleRectangleIntersection(roomCoord,roomWidth,roomLength,roomPosition,roomWidth,roomLength,boundaryForFirstRect=roomBoundary,boundaryForSecondRect=roomBoundary):
                    intersectionWithOtherRoomsExists=True
                    break

        if not receiverHollowRoomCollisionExists and not intersectionWithBlocksExists and not intersectionWithOtherRoomsExists:
            roomPositionsToInstall.append(roomCoord)
            numberOfRoomsCreated+=1
            #print numberOfRoomsCreated

    return np.array(roomPositionsToInstall)


# main function
# strongSignalDistance -> to how many meters we accept this signal as strong. We use it for confidence ellipse calculations
# sensitivityOfResult -> how much sensitive we are about the final position of our object of interest
# maxSignalError -> signals are erronoues in real life, to simulate add noise upto this number
# minUsefulSignal -> min signal value we use for distance calculation
# minSignalValue -> min signal that we can still find, if a signal is lower than that(if receiver is far away), then this receiver(s) cannot catch this signal.
# movingLimit -> how many meters at a time our object moves at max
# movingTendency -> in what direction and meters our object tends to move

def animate_dummy_init():
    pass
def animate(iterNo, ax, macID, currPerson, NumberOfParticles,  xdims=(0, 50), ydims=(0, 50), maxSignalError=20,  movingLimit=2, pastCoeff=0, minUsefulSignal=-90, 
        minSignalValue=-100,numberOfReceivers=4, sensitivityOfResult=1.0, strongSignalDistance=5 , movingTendency=np.array([0,0]) ):


    ax.clear() 
    ax.set_xlim(*xdims)
    ax.set_ylim(*ydims)
    ax.set_aspect('equal',adjustable='box')
    minSideLenghtOfTheMap=np.maximum(xdims[1]-xdims[0],ydims[1]-ydims[0])
    tickStepSize=np.ceil(minSideLenghtOfTheMap/40)
    xstart,xend = ax.get_xlim()
    ystart,yend = ax.get_ylim()
    ax.xaxis.set_ticks(np.arange(xstart, xend+tickStepSize, tickStepSize ))
    ax.yaxis.set_ticks(np.arange(ystart, yend+tickStepSize, tickStepSize ))
    ax.tick_params(axis="x", labelsize=20) 
    ax.tick_params(axis="y", labelsize=20)
    
    roomLineWidth=linewidth_from_data_units(roomWallWidth,ax)
    currPerson.move_beacon_in_map(xdims,ydims,movingLimit,movingTendency,roomBoundary=roomWallWidth/2) # Comment out to if POI position should be entered manually
    #currPerson.beacon_pos = predefinedPos[iterNo] # Remove the comment and fill this variable to determine the POI position for the current time step
    currPerson.calc_RSSIs_to_Receivers(minSignalValue,minUsefulSignal,maxSignalError )
    currPerson.setBeaconDistances_fromRSSIs(minUsefulSignal)

    global numberOfNotFounds
    print iterNo
    isProcessed=False
    if all(dist is None for dist in currPerson.distToReceivers):
        numberOfNotFounds+=1
        pass    
    else:
        currPerson.averaged_beacon_pos = currPerson.multiLateration(xdims,ydims,sensitivityOfResult)


        # 1st STEP
        currPerson.predict_BLE(no_of_noise_elements = NumberOfParticles, movingLimit=movingLimit, pastCoeff = pastCoeff, xdims=xdims, ydims=ydims,movingTendency=movingTendency )           
        # 2nd STEP
        currPerson.update_weights()
        # resample if too few effective particles
        if neff(currPerson.weights) < NumberOfParticles/2.0:
            
            tmp_particles=np.zeros((NumberOfParticles, 2))
            tmp_weights = np.zeros(NumberOfParticles)
            tmp_particles[:]=currPerson.particles[:]
            tmp_weights[:]=currPerson.weights[:]
            currPerson.resample_from_higher_weights(tmp_particles, tmp_weights)         
            if np.allclose(tmp_weights, 1.0/NumberOfParticles):
                currPerson.weights[:]=tmp_weights[:]
                currPerson.particles[:]=tmp_particles[:]
            else:
                #print "no resampling is made for iteration " + iterNo
                pass

        currPerson.calc_PDF(strongSignalDistance,pastCoeff)
        currPerson.prev_covMatrix=currPerson.covMatrix
        
        currPerson.x_pp[:] = currPerson.x_prev[:]      # or np.copyto(x_pp,x_prev)
        currPerson.x_prev[:] = currPerson.particles[:] # or np.copyto(x_prev,particles)

        global OverallError
        CurrAccuracy = np.linalg.norm(currPerson.mu-currPerson.beacon_pos)     
        OverallError += CurrAccuracy
        #if iterNo == totalIterNo-1:
            #print "average Error is: " + str(OverallError/(totalIterNo-numberOfNotFounds) )
            #print "numberOfNotFounds is: " + str(numberOfNotFounds)
        
        particles_x,particles_y=np.hsplit(currPerson.particles,2)
        if not np.isnan(currPerson.covMatrix).any() or \
           not np.isinf(currPerson.covMatrix).any():
            lambda_, v = np.linalg.eig(currPerson.covMatrix)    
            lambda_ = np.sqrt(lambda_)
            # Ellipse drawing code below is borrowed from Jaime's answer in https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib/20127387
            #The following code draws a one, two, and three standard deviation sized ellipses:
            color1,color2,color3=0.0,0.0,0.0
            
            for j in xrange(1, 4):
                ell = Ellipse(xy=(np.mean(particles_x),np.mean(particles_y)),
                              width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                              angle=np.rad2deg(np.arccos(v[0, 0])))
                color1+=0.3
                color2+=0.2
                color3+=0.25
                #ell.set_facecolor((color1, color2, color3))
                ell.set_edgecolor((color1, color2, color3))
                ell.set_fill(False)
                ell.set_linewidth(5.0)

                ax.add_artist(ell)
        else:
            pass # do not draw any ellipses

        # draw particles
        ellipses = [Ellipse(xy=(xi,yi), width=particleWidth, height=particleHeight, linewidth=0, facecolor='black') for xi,yi in zip(currPerson.particles[:, 0],currPerson.particles[:, 1])]
        c = collections.PatchCollection(ellipses)
        ax.add_collection(c)

        muPlot = Ellipse(xy=(currPerson.mu[0],currPerson.mu[1]), width=OOIWidth, height=OOIHeight, linewidth=0, facecolor='purple')
        maxWeightedPlot = Ellipse(xy=(currPerson.max_weighted_particle[0],currPerson.max_weighted_particle[1]), width=OOIWidth, height=OOIHeight, linewidth=0, facecolor='orange')
        ax.add_artist(muPlot)
        ax.add_artist(maxWeightedPlot)

        ax.text(0.3,1.03,s="Current Accuracy is: " + str(float("{0:.2f}".format(CurrAccuracy))) + " m"
           ,horizontalalignment='left' , verticalalignment='bottom' , fontsize=16, transform=ax.transAxes )

        isProcessed = True
 
 
    # draw room, blocks and receivers in this order, since otherwise they may not appear in the map properly 
    if numberOfRooms > 0:
        for roomIndex, roomPosition in enumerate(roomPositions):
            roomBottomLeft=roomPositions-np.array( [roomWidth[roomIndex]/2,roomLength[roomIndex]/2])  
            roomColor=materialColors[ roomMaterials[roomIndex] ]
            ax.add_patch( Rectangle(roomBottomLeft,roomWidth[roomIndex],roomLength[roomIndex],linewidth=roomLineWidth,edgecolor=roomColor,facecolor='None') ) # thich borders without face(inner size) makes a rectangle with a hole
    
    if numberOfBlocks > 0:
        blockBottomLeft=blockPositions-np.array( [blockWidth/2,blockLength/2]) 
        for blockIndex, blockPosition in enumerate(blockBottomLeft):
            blockColor=materialColors[ blockMaterials[blockIndex] ]
            ax.add_patch( Rectangle(blockPosition,blockWidth,blockLength,linewidth=1,edgecolor=blockColor,facecolor=blockColor) )

    if numberOfReceivers > 0:
        receiverBottomLeft=receiverPositions-np.array( [receiverWidth/2,receiverLength/2]) 
        for receiverPosition in receiverBottomLeft:
            ax.add_patch( Rectangle(receiverPosition,receiverWidth,receiverLength,linewidth=1,edgecolor='darkblue',facecolor='darkblue') )


    beaconPosPlot = Ellipse((currPerson.beacon_pos[0],currPerson.beacon_pos[1]), width=OOIWidth, height=OOIHeight, linewidth=0, facecolor='green') 
    ax.add_artist(beaconPosPlot)
    plt.subplots_adjust(left=0,bottom=0.1,right=1,top=0.87,wspace=0,hspace=0)

  
    # Comment out THE FOLLOWING LINES TO DRAW LEGENDS (Purple represents mean of the particles & green represents the BLE Beacon pos)
    if isProcessed:
        ax.legend([beaconPosPlot, muPlot, maxWeightedPlot], ['BLE Beacon Pos', 'Mean Of Particles', 'Most Weighted Particle'], loc="lower left", prop={'size': 10}, bbox_to_anchor=(0, 1))
    else:
        ax.legend([beaconPosPlot], ['BLE Beacon Pos'], loc="lower left", prop={'size': 10}, bbox_to_anchor=(0, 1))

####################################################################################################################################################################################

if __name__ == '__main__':
    main()




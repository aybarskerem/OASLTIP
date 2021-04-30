from __future__ import division

import numpy as np
import math # for math.ceil
import matplotlib.pyplot as plt

from numpy.linalg import norm
from numpy.random import uniform 

from scipy.stats import multivariate_normal # for bivariate gaussian -> brownian motion ( normal with mu x(t-1), and variance sigma )
from filterpy.monte_carlo import systematic_resample, multinomial_resample , residual_resample, stratified_resample
from scipy.optimize import minimize
from scipy.optimize import fmin_tnc
from matplotlib.patches import Ellipse, Rectangle, Circle
import matplotlib.transforms as transforms
from matplotlib import animation
from matplotlib import collections
from numpy.random import seed
from multiprocessing import Process
from collections import deque as col_deque # for the sliding windows
import copy
#from matplotlib.font_manager import FontProperties
import time
from sklearn.cluster import KMeans

from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

#from shapely.geometry.point import Point
import shapely.affinity
import matplotlib.ticker as mticker

from scipy.interpolate import griddata
from scipy.interpolate import interp2d

import matplotlib.patches as mpatches   

from matplotlib import rc
import os
rc('text', usetex=True)

import sys

if len(sys.argv)!=3:
    print "run as 'python <script name> <seed no> <output file name>'"
    exit(1)

# object of interest , all variables used for single object tracking will be used as a member variable
# and all the function will be used as a class function instead of global functions
sizeIncrementRatio=1000/762 # sizeIncrementRatio_small_over_large -> kagida basarken small haritalarin boyutu buyuk oldugundan daha cok resize ediliyor, bunu handle etmeliyiz.

TOTAL_MULTILATERATION_ACCURACY=0
TOTAL_MULTILATERATION_ITER=0



sensitivityOfResult=0.1
maxSignalError=0

numberOfBlocks=0
#blockWidth=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) / 8
#blockLength=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) / 12 
blockWidth=0.5 # 0.7 = 70cm for example
blockLength=2.5

pastCoeff=0.2

totalNumberOfPeople=1
MinWaitingForPerson=0 # min waiting time between each person
MaxWaitingForPerson=20

totalIterNo=6
NumberOfParticles=300
xdims=(0,5)  # our office's coordinates
ydims=(0,3)
#xdims=(0,3) 
#ydims=(0,2)
    
movingLimit=1.0

minUsefulSignal=-90
minSignalValue=-100
numberOfReceivers=3


strongSignalDistance=5
#movingTendency=np.array([0.5,0.2])
movingTendency=np.array([0.0,0.0])

prevMotionRepeatProb=0.75


numberOfRooms=0
#roomWidth=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) / 8
#roomLength=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) / 6
roomWidth=2
roomLength=5
# roomPositions = [ [6.75,7] ]

OOIWidth=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) /10 * sizeIncrementRatio# beacon representing the person is drawn as circle in the map(ellipse indeed, but looks like a circle due to adjustments)
OOIHeight=OOIWidth    

particleWidth=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) /400 * sizeIncrementRatio
particleHeight=particleWidth
# these blocking material positions will be added in main function

# make receivers in square shape
receiverWidth=np.minimum( (xdims[1] - xdims[0]), (ydims[1]-ydims[0] ) ) /30 * sizeIncrementRatio
receiverLength=receiverWidth

receiverPositions=[]
blockPositions=[]
roomPositions=[]

blockMaterials=[]
roomMaterials=[]

WallRoomRatio=0.125# 0/125: rooms have only 1/8 of them as the 2 walls that we intersect(so inner area is 14 wall width totaling 16 wall width area size)
                # distance is already calculated for our RSSI before taking material things into account, so no need to think about empty area in the rooms
roomWallWidth=roomWidth * WallRoomRatio # express line witdht in terms of data points instead of axis 
# since linewidth expand line width towards inside and outside both in equal amount(so roomWallWidth/2 distance check from rectangle boundary is enouhg for collision check)


materials=['concrete']
#materials = ['aluminum','iron', 'concrete', 'brick', 'glass'] # blockMaterials and roomMaterials elements are chosen from this list
materialColors = {'aluminum':'silver','iron':'black', 'concrete':'gray', 'brick':'red', 'glass':'aqua'}  # https://matplotlib.org/users/colors.html
#material_SignalDisturbance_Coefficients={'aluminum':10.0, 'iron':9.0, 'concrete':8.0, 'brick':7.0, 'glass':3.0 } # signal attenuation per 1 meter in terms of dBm
material_SignalDisturbance_Coefficients={'aluminum':20.0, 'iron':18.0, 'concrete':16.0, 'brick':14.0, 'glass':6.0 } # signal attenuation per 1 meter in terms of dBm
smallestFigureSideInInch=6 # smallest side will be 6 inch

TX_Power=0
rssiAtOne=TX_Power-65


fingerPrintingBeaconPositions=np.array( [ [0.25,2.25],  [5,   5   ], [12,   8   ], [11.5, 3   ] ] )
#fingerPrintingBeaconPositions=np.array( [ [0,0],  [5,   5   ], [12,   8   ], [13.5,13   ] ] )
fingerPrintingSignalStrengthBeaconsToReceivers=np.array([ [ -92, -89, -88, -87    ], [ -95, -94, -87, -85   ], [ -88, -91, -78, -79   ] ]) # 4 Beacon to each of the 3 receivers
InterpolatedMapForReceivers=None

interpolatedSignalStrenghForAllPositions_forEachReceiver={} # make it a dictionary where the key is 2d position
useFingerPrinting=True # use fingerprinting instead of multi-laterate , choose the 1st nearest valued position
                
safetyOffset = 10**-10             


OverallError=0       
numberOfNotFounds=0 

#predefinedPos=np.array([ [0.1,0], [0.2,1], [0.22,1.7], [0.3,2.7], [1.5,2.6], [2,1.7], [2.5,0.2], [3.5,0.15] ])
predefinedPos=np.array([ [0.1,0], [0.2,1], [0.22,1.7], [0.3,2.7], [1.5,2.6], [2,1.7] ])

def main():

    global receiverPositions, blockPositions, roomPositions, blockMaterials, roomMaterials, roomWallWidth
    xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]

    #print "processFunction"
    seed(int(sys.argv[1]))
    print "seed is: " + sys.argv[1]

    receiverPositions=getReceiverPositionsToInstall(xdims,ydims,numberOfReceivers)


    blockPositions=getBlockPositionsToInstall(xdims=xdims,ydims=ydims,numberOfBlocks=numberOfBlocks) # install blocks without overlapping
    #roomPositions=getRoomPositionsToInstall(xdims=xdims,ydims=ydims,numberOfRooms=numberOfRooms,roomBoundary=roomWallWidth/2)

    blockMaterials=np.random.choice(materials, numberOfBlocks)
    roomMaterials=np.random.choice(materials, numberOfRooms)

    AllProcesses=[]
    #for i in range(totalNumberOfPeople):

    AllProcesses.append(Process(target=processFunction,args=() ) )

    for proc in AllProcesses: 
        proc.start()
        sleepAmount=np.random.uniform(low=MinWaitingForPerson,high=MaxWaitingForPerson)
        #print "sleepAmount is: " + str(sleepAmount)
        time.sleep(sleepAmount)


def processFunction():
    xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]


    macID=generateRandomMACID()

    while True:
        initialPositionOfThePerson=np.random.uniform(low=[xmin,ymin], high=[xmax,ymax], size=(2))
        isCollision=False
        for blockPosition in blockPositions:
            if checkEllipseRectangleIntersection(initialPositionOfThePerson,OOIWidth,OOIHeight,blockPosition,blockWidth,blockLength):
                isCollision=True
                break
        if not isCollision:
            for roomIndex, roomPosition in enumerate(roomPositions):
                if checkEllipseRectangleIntersection(initialPositionOfThePerson,OOIWidth,OOIHeight,roomPosition,roomWidth,roomLength,boundaryForRect=roomWallWidth/2):
                    isCollision=True
                    break   

        if not isCollision:
            break

    currPerson = OOI(xdims,ydims,NumberOfParticles,receiverPositions,initialPositionOfThePerson)

    iterNo=0
    while iterNo < totalIterNo:
        animate(iterNo, macID, currPerson, NumberOfParticles,xdims,ydims,maxSignalError,movingLimit,pastCoeff,
                                                    minUsefulSignal,minSignalValue,numberOfReceivers,sensitivityOfResult,
                                                    strongSignalDistance,movingTendency)
        iterNo+=1



def checkIfCoordinateIsInMap(coords,width,height):
    xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]
    return coords[0]-width/2 >= xmin and coords[0]+width/2 <= xmax and coords[1]-height/2 >= ymin and coords[1]+height/2 <= ymax


class OOI:
    def __init__(self,xdims,ydims,NumberOfParticles,receiverPositions,initialPositionOfThePerson):

        # INITIALIZATION STEP, distribute particles on the map 
        self.particles = create_uniform_particles(xdims,ydims , NumberOfParticles)
        self.weights = np.ones(NumberOfParticles) / NumberOfParticles
        #beacon_pos = np.array([0.0, 0.0])
        #self.beacon_pos = np.array( [(xdims[1]-xdims[0])/4.0,(ydims[1]-ydims[0])/4.0] )
        self.beacon_pos=initialPositionOfThePerson
        self.prev_walkingNoise=None
        self.x_prev =  np.zeros((NumberOfParticles, 2)) # prev particles
        self.x_pp =  np.zeros((NumberOfParticles, 2)) # prev of prev particle
        self.receiverPositions = receiverPositions
        self.RSSIofReceivers=[] # what are the RSSI valus for this person on our receiver devices
        self.UnprocessedRSSIofReceivers=[] # BLE fingerprinting needs the base value(ham deger) to compare its results with the received(but still weaking due to should be simualated since it is real)
        self.distToReceivers=[]
        self.prevCovMatrix=None
        self.mu=None
        self.max_weighted_particle=None
        self.slidingWindows=[col_deque([]) for i in range(len(receiverPositions) ) ]
        
        
    # circle rectangle detection yapmaliyim aslinda:
    # http://jeffreythompson.org/collision-detection/circle-rect.php
    #ensure person does not go out the map
    # movingLimit is the max step lenght of the person, let's say 1 meter per time step for example
    # movingTendency is the tendency for our the person to move in a direction
    def move_beacon_in_map(self,xdims, ydims,movingLimit,movingTendency=np.array([0,0]),roomBoundary=0 ):

        # hepsini dolassin bloklarin, hicbiri ile kesismiyorsa hareket etsin
        xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]


        xlow = np.maximum(xmin,self.beacon_pos[0]-movingLimit)-self.beacon_pos[0]
        xhigh =np.minimum(xmax, self.beacon_pos[0]+movingLimit)-self.beacon_pos[0]
        ylow = np.maximum(ymin,self.beacon_pos[1]-movingLimit)-self.beacon_pos[1]
        yhigh =np.minimum(ymax, self.beacon_pos[1]+movingLimit)-self.beacon_pos[1]

        while True:
            walking_noise_x = np.random.uniform(low=xlow,high=xhigh) # human motion undeterminism  
            walking_noise_y = np.random.uniform(low=ylow,high=yhigh)

            #walking_noise = np.zeros(particles.shape)
            walkingNoise=np.array( (walking_noise_x,walking_noise_y)).T
            #walkingNoise=np.random.uniform(-movingLimit,movingLimit,size=(2,))

            if self.prev_walkingNoise is not None:
                walkingChoices=[walkingNoise,self.prev_walkingNoise]
                walkingNoise = np.copy(walkingChoices[ np.random.choice([0,1], p=(1-prevMotionRepeatProb,prevMotionRepeatProb)) ] ) # choose the prev motion with a higher probability
           
            tmpBeaconPos=self.beacon_pos + walkingNoise + movingTendency


            #print "beacon pos is: " + str(self.beacon_pos)
            #print "walkingNoise is: " + str(walkingNoise)
            isCollision=not checkIfCoordinateIsInMap(tmpBeaconPos, OOIWidth,OOIHeight)

            if not isCollision:
                for blockPosition in blockPositions:
                    #if checkCircleCollision_WithRectangle(tmpBeaconPos,OOIWidth,OOIHeight,blockPosition,blockWidth,blockLength):
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

    # use constant velocity model described in page 32
    # yurumek icin mu 0.5 metre olur, std ise 0.2m falan. 
    # O zaman variance 0.04 m2 diyebiliriz
    # p(x_t| x{t-1}), su sekilde hesaplanabilir, p(x_t) icin gaussian hesapla, sonra p(x_{t-1} icin hesapla p(x_t,x{t-1} =  p(x_t| x{t-1}) * p(x_{t-1} demek) )
    # 2 prob'un ayni anda bulunmasi demek yani bu 
    # 2 prob'un altta kalan alani, bu area'yi p(x_{t-1}'e bolersek sonucu buluruz) )) -> bu da page 32'deki formule tekabul ediyor(bolmek demek exp'lerin cikarilmasi demek)
    # velocity icin ise x(t-1) ve x(t-2) verilmeli bunlar default'u None olacak, ve eger biri dahi None ise velocity hesaplanamayacagindan ilk oncelerde bunlar
    # hesaba katilamdan prediction yapacagiz, yani brownian motion olmus olacak.
    # 32'deki d'nin ne oldugunu tam anlayamadim, ben 1 kabul edecegim onu direkt olarak.

    # x_prev = x(t-1)
    # x_pp = prev of x_prev
    def predict_BLE( self, no_of_noise_elements, movingLimit, pastCoeff, xdims, ydims, movingTendency=np.array([0,0]) ):

        #rand_gaussian_noise=np.random.multivariate_normal(mu=mu,cov=sigma,size=no_of_noise_elements) # Draw random samples from a multivariate normal distribution
        #rand_gaussian_noise = 0

        xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]
        # ALL PARTICLES SHOULD RESIDE IN THE MAP, CHECK FOR BEING INSIDE FOR EACH PARTICLE (MOVE THAT AMOUNT AT THE BORDERS AT MAX)

        # min of x, should not be lower than map's xmin && max of x should not be larger than map's xmax
        # meaning low should be max(xmin,particles[:,0]-xmin-movingLimit) && high = min(xmax, xmax-particles[:,0]+movingLimit)
        xlow = np.maximum(xmin,self.particles[:,0]-movingLimit)-self.particles[:,0]
        xhigh =np.minimum(xmax, self.particles[:,0]+movingLimit)-self.particles[:,0]
        ylow = np.maximum(ymin,self.particles[:,1]-movingLimit)-self.particles[:,1]
        yhigh =np.minimum(ymax, self.particles[:,1]+movingLimit)-self.particles[:,1]
        walking_noise_x = np.random.uniform(low=xlow,high=xhigh,size=self.particles.shape[0]) # human motion undeterminism  
        walking_noise_y = np.random.uniform(low=ylow,high=yhigh,size=self.particles.shape[0])
        ##print "walking_noise_x is: " + str(walking_noise_x)
        #walking_noise = np.zeros(particles.shape)
        walking_noise_x=np.array(walking_noise_x)
        walking_noise_y=np.array(walking_noise_y)
        walking_noise=np.array( (walking_noise_x,walking_noise_y)).T 
       
        if np.count_nonzero(self.x_prev) != 0 and np.count_nonzero(self.x_pp) != 0:
            past_velocity = self.x_prev - self.x_pp 

            change_in_pos = (1-pastCoeff) * walking_noise + pastCoeff * past_velocity  # constant_velocity_motion
        else:
            change_in_pos = walking_noise
        #particles += 
        self.particles += change_in_pos + movingTendency    

    # Update the weight of the particles according to the measured beacon position found in the multilateration algorithm for the current time step
    def update_weights(self):
        distances = np.linalg.norm(self.particles - self.averaged_beacon_pos, axis=1)

        self.weights *= np.sum(distances)/distances
        

        # SET ALL WEIGHTS INTERSECTING WITH AN OBSTRUCTION TO ZERO (so that particles do not accumulate on obstructions)
        for particleIndex, particle in enumerate(self.particles):
            isCollision=False
            for blockPosition in blockPositions:
                #if checkCircleCollision_WithRectangle(tmpBeaconPos,OOIWidth,OOIHeight,blockPosition,blockWidth,blockLength):
                if checkEllipseRectangleIntersection(particle,particleWidth,particleHeight,blockPosition,blockWidth,blockLength):
                    isCollision=True
                    break
            if not isCollision:
                for roomIndex,roomPosition in enumerate(roomPositions):
                    #if checkCircleCollision_WithRectangle(tmpBeaconPos,beaconRadius,roomPosition,roomWidth[roomIndex],roomLength[roomIndex]):
                    #print "room wall width is: " + str(roomWallWidth)
                    # use roomWallWidth/2, since linewidth expands toward outside and inside (for roomWallWidth, expands roomWallWidth/2 towards inside and roomWallWidth/2 towards outside)
                    if checkEllipseRectangleIntersection(particle,particleWidth,particleHeight,roomPosition,roomWidth[roomIndex],roomLength[roomIndex],boundaryForRect=roomWallWidth[roomIndex]/2):
                        isCollision=True
                        break 
            if isCollision:
                self.weights[particleIndex]=0
        self.weights += 10**(-300)       # avoid round-off to zero
        self.weights /= sum(self.weights) # normalize

    # Resample N_eff
    def resample_from_higher_weights(self,tmp_particles, tmp_weights):
        #indices = multinomial_resample(weights)
        #indices = residual_resample(weights)
        #indices = stratified_resample(weights)
        indices = systematic_resample(self.weights)

        tmp_particles[:] = tmp_particles[indices]
        tmp_weights[:] = tmp_weights[indices]
        tmp_weights.fill(1.0 / len(tmp_weights))


    # maxSignalError in dBm
    # it should call checkLineSegmentCollision_WithRectange, to lower signal if receiver and beacon is not in "Line of Sight" 
    def calc_RSSIs_to_Receivers(self,minSignalValue,minUsefulSignal,maxSignalError):

        self.RSSIofReceivers[:] = []
        self.UnprocessedRSSIofReceivers[:] = []
        receiverIndex=0
        for receiverPosition in self.receiverPositions:
            res_unprocessedRSSI = 0
            if(maxSignalError > 0):
                res_unprocessedRSSI=weakenedSignal( distance_to_RSSI( np.linalg.norm(receiverPosition-self.beacon_pos) ) , maxSignalError ) 
            else:
                ##print "the norm is: " + str(np.linalg.norm(receiverPosition-self.beacon_pos ))
                res_unprocessedRSSI=distance_to_RSSI( np.linalg.norm(receiverPosition-self.beacon_pos ) )
            #return max(-100,unprocessedRSSI) # Generally signals lower than -100 are not that reliable

            isCollision=False
            

            # this is used to weaken the signal in case there was a block or room between the receiver and the beacon(this is real calculation)
            # this simulates the signal before we catch it in real life.
            
            weakeningAmount=0 # distance between the receiver and the beacon / 1 meter *  ( how many dBm to reduce for 1 meter)
            for blockIndex, blockPosition in enumerate(blockPositions):
                receiverBeaconBlockIntersection=findRectangleLineSegmentIntersectionPoints(receiverPosition,self.beacon_pos,blockPosition,blockWidth,blockLength)
                
                if receiverBeaconBlockIntersection is not None:
                    #print "receiverBeaconBlockIntersection" + str(receiverBeaconBlockIntersection)
                    isCollision=True
                    weakeningAmount+=np.linalg.norm(receiverBeaconBlockIntersection[0,:]-receiverBeaconBlockIntersection[1,:]) * material_SignalDisturbance_Coefficients[ blockMaterials[blockIndex] ] * np.random.uniform(0.5,1.5)  # +- some noise olsun
                    # her engel icin noise eklemek, gercek hayat icin de uygun olacaktir
                    # aslinda burada duvarin material'i ile de hareket etmeliyim. Coefficient'lar 1m icin idi sonucta
                    # distance/1 * coefficient , yani distance(in meters) * coefficient olmali

            
            for roomIndex, roomPosition in enumerate(roomPositions):
                receiverBeaconRoomIntersection=findRectangleLineSegmentIntersectionPoints(receiverPosition,self.beacon_pos,roomPosition,roomWidth,roomLength)
                if receiverBeaconRoomIntersection is not None:
                    #print "receiverBeaconRoomIntersection" + str(receiverBeaconRoomIntersection)
                    isCollision=True
                    weakeningAmount+=np.linalg.norm(receiverBeaconRoomIntersection[0,:]-receiverBeaconRoomIntersection[1,:]) * WallRoomRatio * material_SignalDisturbance_Coefficients[ roomMaterials[roomIndex] ] * np.random.uniform(0.5,1.5)
                    # * some coefficient(odada cok zayiflamasin), odadaki duvarlar kadar zayiflasa yeterli bu da odanin 8'de biri kadar falan olur(kestigimiz 2 duvari da dusunecek olursak)  



            strengtheningAmount=0 # (distance between the receiver and the mean of the particles) / 1 meter *  ( how many dBm to reduce for 1 meter)
            # the calculations below are not real. They are our prediction by looking at the mean value of the particles
            # if the mean of the prev calculations and the beacons have a block or room in between, we better increase the signal
            # this simulates after receiving the signal in real life (post processing of the signal)
            isMeanReceiverCollision=False # this is used to strengthen the received signal in case there was a block in between previously
            if self.mu is not None: #!= degildir kulanma cunku, array oldugu zaman burasi hata verir
                for blockIndex, blockPosition in enumerate(blockPositions):
                    receiverMeanBlockIntersection = findRectangleLineSegmentIntersectionPoints(receiverPosition,self.mu,blockPosition,blockWidth,blockLength)
                    if receiverMeanBlockIntersection is not None:
                        #print "receiverMeanBlockIntersection" + str(receiverMeanBlockIntersection)
                        isMeanReceiverCollision=True
                        strengtheningAmount+=np.linalg.norm(receiverMeanBlockIntersection[0,:]-receiverMeanBlockIntersection[1,:]) * material_SignalDisturbance_Coefficients[ blockMaterials[blockIndex] ] 
                for roomIndex, roomPosition in enumerate(roomPositions):
                    receiverMeanRoomIntersection = findRectangleLineSegmentIntersectionPoints(receiverPosition,self.mu,roomPosition,roomWidth,roomLength)
                    if receiverMeanRoomIntersection is not None:
                        #print "receiverMeanRoomIntersection" + str(receiverMeanRoomIntersection)
                        isMeanReceiverCollision=True
                        strengtheningAmount+=np.linalg.norm(receiverMeanRoomIntersection[0,:]-receiverMeanRoomIntersection[1,:]) * WallRoomRatio * material_SignalDisturbance_Coefficients[ roomMaterials[roomIndex] ]   

            if isCollision:
                ##print "No Line Of Sight between receiver " + str(receiverPosition) + " and beacon " + str(self.beacon_pos)
                #res_unprocessedRSSI=( weakenedSignal(res_unprocessedRSSI,maxSignalError) + res_unprocessedRSSI ) / 2.0 #weaken a bit, but not weaken upto max signal error
                res_unprocessedRSSI-=weakeningAmount
            else:
                pass
                ##print "Direct Line Of Sight between receiver " + str(receiverPosition) + " and beacon " + str(self.beacon_pos) 

            res_processedRSSI=res_unprocessedRSSI

            if isMeanReceiverCollision:
                res_processedRSSI+=strengtheningAmount
                ##print "increased signal strength since there was a wall between the receiver and the beacon in the previous step according to our particle calculations"


            # ONE MORE CHECK FOR SLIDING WINDOWS #

            # each receiver should have a sliding window
            # max slidingWindows size should be 7
            slidingWindow = self.slidingWindows[receiverIndex]
            while len(slidingWindow) >=7:
                ##print "prev size of the window is: " + str( len(self.slidingWindows) )
                slidingWindow.popleft() # delete oldest element
            ##print "after size of the window is: " + str( len(self.slidingWindows) )
            slidingWindow.append(res_processedRSSI) # appends at the right
            ##print "final size of the window is: " + str( len(self.slidingWindows) )


            if self.filterAndCheckSignal(minUsefulSignal,receiverIndex) and res_processedRSSI > minSignalValue:
                ##print "filtering was successful"
                self.RSSIofReceivers.append( res_processedRSSI )
                self.UnprocessedRSSIofReceivers.append( res_unprocessedRSSI )
            else:
                ##print "filtering was not successful"
                self.RSSIofReceivers.append( None )
                self.UnprocessedRSSIofReceivers.append( None )

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


    # if RSSI is lower than -90dBm , then omit this receiver ( assuming we use 0dBm signal powered beacons)
    def setBeaconDistances_fromRSSIs(self,minUsefulSignal):
        self.distToReceivers[:] = []
        for RSSIofReceiver in self.RSSIofReceivers:
            #print "rssi of receiver is: " + str(RSSIofReceiver)
            if RSSIofReceiver is not None and  \
            RSSIofReceiver > minUsefulSignal:
                self.distToReceivers.append( RSSI_to_distance( RSSIofReceiver ) + safetyOffset ) # add safetyOffset0 to avoid divide by zero in the custom_minimize function
            else:
                self.distToReceivers.append( None )

    # NumberOfParticles for 4 RECEIVER
    def multiLateration(self,xdims,ydims,sensitivityOfResult):

        receiverPositionsArray=np.array(self.receiverPositions)
        ##print "elements are : " + str( elements )
        #resultingPoint = Trilaterate(rp1.coord,elements[0],rp2.coord,elements[1],rp3.coord,elements[2]) 
        #resultingPoint = minimize_dist_error(elements,np.vstack(coordinates ),xdims,ydims )

        #with open('deneme.txt', 'a') as the_file:
        #    the_file.write("beacon_pos is: " + str(self.beacon_pos) + "\n" )
        #print "beacon_pos is: " + str(self.beacon_pos)
        # if checkForBlocks == True, it also considers blocks for minimization in the disadvantage of time consumption
        # checkForBlocks means include None info to make multi lateration calculations
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

        """returns mu and variance of the weighted particles"""
        self.mu = np.average(self.particles, weights=self.weights, axis=0)
        #var  = np.average((particles - mu)**2, weights=weights, axis=0)
        self.covMatrix = np.cov(m=self.particles, rowvar=False, aweights=self.weights) # rowvar has to be False otherwise each row represents a variable, with observations in the columns. 
                                                                  # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.cov.html
        self.max_weighted_particle =  self.particles[np.argmax(self.weights) ]

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
            #confidenceEllipseMultiplier=float("inf") # boyle olunca hic cizmesin ellipse
            confidenceEllipseMultiplier=10.0 # 10.0 max'imiz olsun mesela

        self.covMatrix*=confidenceEllipseMultiplier

        # if pastCoeff == 1, o zaman ilk tur harici covMatrix hep prev'e esit olacak. Yani ilk turda buldugu covariance hep esas algidi olmus olacak
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
        #print "ellipse line intersection is: " + str(intersectionPoint)
        #intersectionPoint=np.asarray(intersectionResult.geoms[0].coords[0],intersectionResult.geoms[1].coords[0])
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

    #print "bottomLeftCorner is: " + str(bottomLeftCorner)
    #print "topRightCorner is: " + str(topRightCorner)
    #print "room position is " + str(rectCenter)


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
# if line intersects the rectangle only at 1 point(which could be the rectangle's corner, then we can return None since there is almost no intersection)
# it may intersect at infinite points for points in the same line with an edge(but since it only)
# bunun disinda x,y kontrolu yaparken odanin icinde hic olma ihtimallerini dusurecek cunku oda icindeki point'ler hep kesiiyor olacak ne kadar kalinligi varsa artik
# aslinda odanin icine girme mevzusunu bu handle etmiyor(cunku odanin icine giremiyor benim yesil beacon'im ama girebilse intersection kontrolu oda icerisinde olmamali)
# aslidna line segment yani x,y oda icerisinde ise hic kabul etmemeli bu x,y'i(simulasyon geregi burada olamaz ama simdilik odalara girilmiyor diye kabul ediyorum)
# simdilik oda icerisindeki noktalar az da olsa cezalandiriliyor boyle kalsin artik cok yakinlasamayacagimiz icin zaten buradaki noktalarin sansi dusuk
# sonsuz intersection ve tekli intersecitno'lari engellesek yeter simdilik
# aslinda contains de sonsuz noktada kesiiyor demek, demek ki sonsuz nokta kesisiminde line'in kesisen ilk ve son noktalarini veriyor
# belki de rectangle'in icini bos kabul ediyor, odanin icerisindekileri de hic cezalandirmiyoruz bilemiyorum
def findRectangleLineSegmentIntersectionPoints(p1,p2,rectCenter,rectWidth,rectLength,boundaryForRect=0):
    # CORNERS
    if np.array_equal(p1,p2):
        return None

    bottomLeftCorner=rectCenter+np.array([-(rectWidth/2 + boundaryForRect),-(rectLength/2 + boundaryForRect) ])
    topLeftCorner=rectCenter+np.array([-(rectWidth/2 + boundaryForRect) ,rectLength/2 + boundaryForRect])

    bottomRightCorner=rectCenter+np.array([rectWidth/2 + boundaryForRect,-(rectLength/2 + boundaryForRect) ])
    topRightCorner=rectCenter+np.array([rectWidth/2 + boundaryForRect,rectLength/2 + boundaryForRect])

    line = LineString([p1,p2])
    rectangle = Polygon([bottomLeftCorner, topLeftCorner, topRightCorner, bottomRightCorner])

    #print "findRectangleLineSegmentIntersectionPoints"

    if rectangle.intersects(line):
        intersectionPointObject = rectangle.intersection(line)
        #print intersectionPointObject.coords[0]
        #print intersectionPointObject.coords[1]

        #print np.array(intersectionPointObject.coords).shape
        if np.array_equal(np.array(intersectionPointObject.coords).shape,np.array([2, 2])):
            intersectionPoint=np.array([intersectionPointObject.coords[0],intersectionPointObject.coords[1]])
        else:
            intersectionPoint=None
        #print "rectangle line intersection is: " + str(intersectionPoint)
        #intersectionPoint=np.asarray(intersectionResult.geoms[0].coords[0],intersectionResult.geoms[1].coords[0])
    else:
        intersectionPoint=None


    return intersectionPoint


def generateRandomMACID():
    return ':'.join('%02x'%np.random.randint(0,256) for _ in range(6))

# zayiflamasina ragmen bir istasyona gelen sinyal guclu ise, bu sinyal diger zayif sinyallerden daha degerli
# bu yukaridazden distToReceivers degeri kucuk olan bir sinyal bizim icin daha cok anlama ifade ediyor
# bu degeri kucuk olan istedigimiz icin bu deger ile carparsam o sum daha kucuk olur, bizim istedigimizi elde ihtimalimiz artar

# multilateratiion icin check edecegimiz [x,y] noktasi, eger ble fingerprinting result'taki ile +-2dBm'den fark ediyorsa cezalandir. Bu noktalarin olma ihtimali daha az cunku 
def custom_minimize(RSSIofReceivers, receiverPositions,xdims,ydims,sensitivityOfResult=1.0,checkForBlocks=True):
    mysum=float("inf")
    maxCatchableSignalDistance = RSSI_to_distance( minUsefulSignal ) + safetyOffset
    #print "maxCatchableSignalDistance is: " + str(maxCatchableSignalDistance)
    resultingPoint=[-1,-1]
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
                continue # this point cannot be what we are looking for      
            
            tmp_sum=0
            for i in range(len(receiverPositions)):  

                strengtheningAmount=0
                for blockIndex, blockPosition in enumerate(blockPositions):  # it will not enter this loop if there are no blocks
                    receiverMeanBlockIntersection = findRectangleLineSegmentIntersectionPoints(receiverPositions[i],np.array([x,y]),blockPosition,blockWidth,blockLength)
                    if receiverMeanBlockIntersection is not None:
                        #print "receiverMeanBlockIntersection" + str(receiverMeanBlockIntersection)
                        strengtheningAmount+=np.linalg.norm(receiverMeanBlockIntersection[0,:]-receiverMeanBlockIntersection[1,:]) * material_SignalDisturbance_Coefficients[ blockMaterials[blockIndex] ] 
                for roomIndex, roomPosition in enumerate(roomPositions):
                    # when tryin all possible x and y, this x and y should not be equal to the receivers position, since it would not be a line
                    # if it is equal to the receivers position, the intersection should return None
                    # so findRectangleLineSegmentIntersectionPoints function should return None if points to make the lines are equal
                    # also if intersection is at a corner(which means intersect only at 1 point, then it should return None for this case as well since intersection dist would be zero already)
                    receiverMeanRoomIntersection = findRectangleLineSegmentIntersectionPoints(receiverPositions[i],np.array([x,y]),roomPosition,roomWidth[roomIndex],roomLength[roomIndex])
                    if receiverMeanRoomIntersection is not None:
                        #print "receiverMeanRoomIntersection" + str(receiverMeanRoomIntersection)
                        strengtheningAmount+=np.linalg.norm(receiverMeanRoomIntersection[0,:]-receiverMeanRoomIntersection[1,:]) * WallRoomRatio * material_SignalDisturbance_Coefficients[ roomMaterials[roomIndex] ]   


                xyDistToRec = np.linalg.norm( [x,y] - receiverPositions[i] )
                if RSSIofReceivers[i] is not None:
                    distToReceiverGivenRSSI=RSSI_to_distance( RSSIofReceivers[i] + strengtheningAmount) + safetyOffset

                    tmp_sum+=( abs( xyDistToRec - distToReceiverGivenRSSI ) / distToReceiverGivenRSSI ) ** 2 

                # eger 5 turdur arka arkaya None ise yapsin hemen None ise degil -> zaten sinyalleri buraya gondermeden ona gore ayarliyorum

                else: #  distToReceivers[i] None ise, [x,y]'intersection receiver'imiza belli yakinliktan fazla yakin olmasi imkansiz olmali(bundan daha yakin ise cezalandir)
                        # [x,y], receiverPositions[i]'ye ne kadar yakinda o kadar cezalandir
                        # distToReceivers[i] bizim belirledigimiz bir sey zaten tahminimiz yani. Biz bunun yerine mesela 10m koyabiliriz bundan ne kdar deviate etmis diye
                        # ama bizim icin ne kadar yakinda o kadar kotu cunku biz belirli bir uzaklik tahmin ediyoruz, o yuzden 1/distToReceivers yerine 1/ ( [x,y]-receiverPositons) koyalim
            
                    #if checkForBlocks:                        
                    maxCatchableSignalDistance = RSSI_to_distance( minUsefulSignal + strengtheningAmount) + safetyOffset
                    if xyDistToRec < maxCatchableSignalDistance: # we see it as None, so it should not be closer than maxCatchableSignalDistance. If so, then punish
                        tmp_sum+=( abs( xyDistToRec - maxCatchableSignalDistance )  / xyDistToRec ) ** 2

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


# for each receiver hold a separate signal strenght map
# each beacon should have its interpolation all around the map. Then we we should take weighted average of these beacons signal strengths values
# For example, FOR RECEIVER 1,  if beacon1 is at [5,5] and beacon2 is at [10,3] and the point that we want to interpolate is at [10,5]. Beacon2 should have higher vote to determine signal strength
# signal strength values of the beacons (fingerpritn positions) are different for each receiver, therefore for each receiver we should hold another map info
def interpolateFingerPrintingResult():
    xElems=np.arange(xdims[0],xdims[1],sensitivityOfResult)
    yElems=np.arange(ydims[0],ydims[1],sensitivityOfResult  )
 
    allPosDistancesToReceivers={} # make it a dictionary where the key is 2d position
    for i in range(numberOfReceivers):
        for x in xElems:
            for y in yElems:
                allPosDistancesToReceivers[i,x,y]=np.linalg.norm(receiverPositions[i]- [x,y])

    numberOfBeacons=fingerPrintingSignalStrengthBeaconsToReceivers.shape[1]
    allPosDistancesToBeacons={} # make it a dictionary where the key is 2d position
    for k in range(numberOfBeacons):
        for x in xElems:
            for y in yElems:
                allPosDistancesToBeacons[k,x,y]=np.linalg.norm(fingerPrintingBeaconPositions[k]- [x,y])

    

    # INITIALIZE INTERPOLATION MAP FOR EACH RECEIVER
    global interpolatedSignalStrenghForAllPositions_forEachReceiver
    for i in range(numberOfReceivers):
        for x in xElems:
            for y in yElems:
                interpolatedSignalStrenghForAllPositions_forEachReceiver[i,x,y]=0


    for i in range(numberOfReceivers):
        for x in xElems:
            for y in yElems:
                minDist=np.float('inf')
                min_k=0
                # find the closest beacon to [x,y]
                for k in range(numberOfBeacons):
                    if allPosDistancesToBeacons[k,x,y] < minDist:
                        min_k=k
                        minDist = allPosDistancesToBeacons[k,x,y]

                base_dist=np.linalg.norm(fingerPrintingBeaconPositions[min_k]-receiverPositions[i])
                target_dist=allPosDistancesToReceivers[i,x,y]
                base_RSSI=fingerPrintingSignalStrengthBeaconsToReceivers[i][min_k]
                # whichever beacon or receiver is the closest to [x,y], it should determine the interpolation result
                # yada receiver'lar daha yakin ise o noktalara 0 olarak vs. kalsin
                # en sonra da buradaki tahmini degerleri hic bir blok yokmuscasina receiver versin
                interpolatedSignalStrenghForAllPositions_forEachReceiver[i,x,y]+=calc_relative_RSSI(base_dist,target_dist,base_RSSI)
                print calc_relative_RSSI(base_dist,target_dist,base_RSSI)


    print interpolatedSignalStrenghForAllPositions_forEachReceiver


def calc_relative_RSSI(base_dist, target_dist, base_RSSI):
    if target_dist >= 1:
        return base_RSSI + -20 * np.log ( (target_dist) / (base_dist+safetyOffset) )
    else:
        return zero_one_meter_distance_to_RSSI(target_dist)

#distance in meters, returns RSSI in dBm
# assuming signal propogation constant is 2, https://www.rn.inf.tu-dresden.de/dargie/papers/icwcuca.pdf in equation (4)
# distance 4'den 8'e cikinca 0.6'dan 0.9'a cikiyor(negative ile carpildigi icin output), output daha az azalmis oluyro dist arttikca
# zero_one_meter_distance_to_RSSI'te ise mesela dist 0.1'den 0.2'ye ciksa sonuc 0.15'en 0.34'e cikiyor -> yani rssi daha hizli azalmis oluyor 
def distance_to_RSSI(distance):
    res_RSSI = 0
    ##print "distance is: " + str(distance)
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
    #return float( np.log( (np.e - 1)/rssiAtOne * RSSI + 1 ) ) 
    return 10**( ( ( RSSI - TX_Power ) * np.log10(2) ) / (rssiAtOne - TX_Power) )  -1

# should return something between TX power and rssiAtOne
def zero_one_meter_distance_to_RSSI  (dist):
    #return float( rssiAtOne * ( (np.exp(dist) - 1) / (np.e - 1) ) )
    return float( TX_Power + (rssiAtOne - TX_Power) * ( (np.log10(dist+1)) / (np.log10(2) ) ) )
    #float( (1-dist)*TX_Power + dist*rssiAtOne


# N_eff : Effective weight number
def neff(weights):
    return 1.0 / np.sum(np.square(weights))


def getReceiverPositionsToInstall(xdims,ydims,numberOfReceivers):
    xmin,xmax,ymin,ymax= xdims[0],xdims[1],ydims[0],ydims[1]
    #reaOfTheMap=int( (ymax-ymin)*(xmax-xmin) )
    step_size=(1/( np.ceil(np.sqrt(numberOfReceivers*1000) ) ) )
    while True:
        #initial_points=np.random.uniform(low=[xmin,ymin], high=[xmax,ymax], size=(areaOfTheMap*2,2)) # I deleted .tolist()
        #x_step_size=(xdims[1]-xdims[0])/3
        #y_step_size=(ydims[1]-ydims[0])/3

        #print "step_size is: " + str(step_size)
        initial_points = np.mgrid[0:1+step_size:step_size, 0:1+step_size:step_size].reshape(2,-1).T

        receiverPositions = KMeans(n_clusters=numberOfReceivers, random_state=0,n_init=100).fit(initial_points).cluster_centers_
        #receiverPositions=kmeans(initial_points,numberOfReceivers)
        if receiverPositions is not None:
            ##print "initial receiver positions area " + str(receiverPositions)
            receiverPositions[:,0]=xmin+receiverPositions[:,0]*(xmax-xmin)
            receiverPositions[:,1]=ymin+receiverPositions[:,1]*(ymax-ymin)
            ##print "after receiverPositions are " + str(receiverPositions)
        return receiverPositions
            #return initial_points    


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
            #print numberOfBlocksCreated

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
def animate(iterNo, macID, currPerson, NumberOfParticles,  xdims=(0, 50), ydims=(0, 50), maxSignalError=20,  movingLimit=2, pastCoeff=0, minUsefulSignal=-90, 
        minSignalValue=-100,numberOfReceivers=4, sensitivityOfResult=1.0, strongSignalDistance=5 , movingTendency=np.array([0,0]) ):

    
    print "iterNo is: ", iterNo
    currPerson.move_beacon_in_map(xdims,ydims,movingLimit,movingTendency,roomBoundary=roomWallWidth/2)
    #currPerson.beacon_pos = predefinedPos[iterNo]
    currPerson.calc_RSSIs_to_Receivers(minSignalValue,minUsefulSignal,maxSignalError )
    currPerson.setBeaconDistances_fromRSSIs(minUsefulSignal)

    global numberOfNotFounds
    print iterNo
    isProcessed=False
    if all(dist is None for dist in currPerson.distToReceivers):
        #print "all distances are None, no processing"
        numberOfNotFounds+=1
        pass    
    else:
        currPerson.averaged_beacon_pos = currPerson.multiLateration(xdims,ydims,sensitivityOfResult)

        #print "real pos is: " + str(currPerson.beacon_pos)
        #print "multilateratiion pos is: " + str(currPerson.averaged_beacon_pos)
        
        #print "averaged_beacon_pos for " + macID + " is: " + str(currPerson.averaged_beacon_pos)
        #print "the real pos for " + macID + " is: "  + str(currPerson.beacon_pos)

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
        
        # https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
        
        particles_x,particles_y=np.hsplit(currPerson.particles,2)
  

    if iterNo == totalIterNo-1:
        #print "OverallError error is: " + str(OverallError)
        #print "average Error is: " + str(OverallError/(totalIterNo-numberOfNotFounds) )
        #print "numberOfNotFounds is: " + str(numberOfNotFounds)
        with open(sys.argv[2]+"_avgError.txt","a+") as outFile:
            if OverallError!=0:
                outFile.write(str(OverallError/(totalIterNo-numberOfNotFounds) ) + "\n")
            else:
                outFile.write(str(OverallError) + "\n")
        
        with open(sys.argv[2]+"_noSignalError.txt","a+") as outFile:
            outFile.write(str(numberOfNotFounds) + "\n" )
    #if isProcessed:
    #    ax.legend([beaconPosPlot, muPlot, maxWeightedPlot], ['BLE Beacon Pos', 'Mean Of Particles', 'Most Weighted Particle'], loc="lower left", prop={'size': 10}, bbox_to_anchor=(0, 1))
    #else:
    #    ax.legend([beaconPosPlot], ['BLE Beacon Pos'], loc="lower left", prop={'size': 10}, bbox_to_anchor=(0, 1))

####################################################################################################################################################################################

if __name__ == '__main__':
    main()




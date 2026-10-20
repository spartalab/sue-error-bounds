class Zone:
    def __init__(self, _tmpIn):
        self.zoneId = _tmpIn[0]
        self.lat = 0
        self.lon = 0
        self.destList = []


class Node:
    '''
    This class has attributes associated with any node
    '''
    def __init__(self, _tmpIn):
        self.Id = _tmpIn[0]
        self.lat = 0
        self.lon = 0
        self.outLinks = []
        self.inLinks = []
        self.label = float("inf")
        self.pred = ""
        self.inDegree = 0
        self.outDegree = 0
        self.order = 0 # Topological order
        self.wi = 0.0 # Weight of the node in Dial's algorithm
        self.xi = 0.0 # Toal flow crossing through this node in Dial's algorithm


class Link:
    '''
    This class has attributes associated with any link
    '''
    def __init__(self, _tmpIn):
        self.tailNode = _tmpIn[0]
        self.headNode = _tmpIn[1]
        self.capacity = float(_tmpIn[2]) # veh per hour
        self.length = float(_tmpIn[3]) # Length
        self.fft = float(_tmpIn[4]) # Free flow travel time (min)
        self.beta = float(_tmpIn[6])
        self.alpha = float(_tmpIn[5])
        self.speedLimit = float(_tmpIn[7])
        self.tollInTime = 0.0
        #self.toll = float(_tmpIn[9])
        #self.linkType = float(_tmpIn[10])
        # self.flow = 0.0
        # self.cost =  float(_tmpIn[4])*(1 + float(_tmpIn[5])*math.pow((float(_tmpIn[7])/float(_tmpIn[2])), float(_tmpIn[6])))
        # self.logLike = 0.0
        # self.reasonable = True # This is for Dial's stochastic loading
        # self.wij = 0.0 # Weight in the Dial's algorithm
        # self.xij = 0.0 # Total flow on the link for Dial's algorithm


class Demand:
    def __init__(self, _tmpIn):
        self.fromZone = _tmpIn[0]
        self.toNode = _tmpIn[1]
        self.demand = float(_tmpIn[2])


class Network:
    def __init__(self):
        self.tripSet = {}
        self.zoneSet = {}
        self.nodeSet = {}
        self.linkSet = {}

    def readDemand(self, inputLocation):
        inFile = open(inputLocation+ "demand.dat")
        tmpIn = inFile.readline().strip().split("\t")
        for x in inFile:
            tmpIn = x.strip().split("\t")
            self.tripSet[tmpIn[0], tmpIn[1]] = Demand(tmpIn)
            if tmpIn[0] not in self.zoneSet:
                self.zoneSet[tmpIn[0]] = Zone([tmpIn[0]])
            if tmpIn[1] not in self.zoneSet:
                self.zoneSet[tmpIn[1]] = Zone([tmpIn[1]])
            if tmpIn[1] not in self.zoneSet[tmpIn[0]].destList:
                self.zoneSet[tmpIn[0]].destList.append(tmpIn[1])

        inFile.close()
        print(len(self.tripSet), "OD pairs")
        print(len(self.zoneSet), "zones")

    def readNetwork(self, inputLocation):
        inFile = open(inputLocation + "network.dat")
        tmpIn = inFile.readline().strip().split("\t")
        for x in inFile:
            tmpIn = x.strip().split("\t")
            self.linkSet[tmpIn[0], tmpIn[1]] = Link(tmpIn)
            if tmpIn[0] not in self.nodeSet:
                self.nodeSet[tmpIn[0]] = Node(tmpIn[0])
            if tmpIn[1] not in self.nodeSet:
                self.nodeSet[tmpIn[1]] = Node(tmpIn[1])
            if tmpIn[1] not in self.nodeSet[tmpIn[0]].outLinks:
                self.nodeSet[tmpIn[0]].outLinks.append(tmpIn[1])
            if tmpIn[0] not in self.nodeSet[tmpIn[1]].inLinks:
                self.nodeSet[tmpIn[1]].inLinks.append(tmpIn[0])

        inFile.close()
        print(len(self.nodeSet), "nodes")
        print(len(self.linkSet), "links")


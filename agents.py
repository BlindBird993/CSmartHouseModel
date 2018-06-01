from mesa import Agent, Model
import random
import copy
import getDataFromExcel as data
import numpy as np
import collections as col
import scipy.stats as sts
import operator


WIND_DATA = data.getData()

class ControlAgent(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)

        self.hour = 0
        self.day = 0
        self.week = 0

        self.buyers = []
        self.sellers = []

        self.stepCount = 0

        self.demands = []
        self.productions = []

        self.historyDemands = []
        self.historyProductions = []

        self.distributedDemands = []
        self.summedDemands = []

        self.boughtFromTheGrid = []

        self.demandPrice = []
        self.supplyPrice = []

        self.clearPrice = 0
        self.surplus = 0

        self.totalDemand = 0
        self.totalSupply = 0

        self.totalProduction = 0
        self.listOfConsumers = None
        self.dictOfConsumers = None

        self.numberOfBuyers = 0
        self.numberOfSellers = 0

        self.numberOfConsumers = 0

    def getTotalSupply(self):
        self.totalSupply = 0.0
        for agent in self.model.schedule.agents:
            if (isinstance(agent, SolarPanelAgent) or isinstance(agent, StorageAgent) or isinstance(agent,WindEnergyAgent)):
                if agent.readyToSell is True:

                    self.totalSupply = round(self.totalSupply + agent.energy, 3)
        self.historyProductions.append(self.totalSupply)
        print("Total Supply {}".format(self.totalSupply))

    def getTotalDemand(self):
        self.totalDemand = 0.0
        for agent in self.model.schedule.agents:
            if (isinstance(agent, HeaterAgent) or isinstance(agent, StorageAgent) or isinstance(agent,
                                                                                                HeatedFloorAgent)):
                if agent.readyToBuy is True:

                    self.totalDemand = round(self.totalDemand + agent.currentDemand, 3)
        self.historyDemands.append(self.totalDemand)
        print("Total Demand {}".format(self.totalDemand))

    def getConsumerDict(self):
        self.listOfConsumers = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent, HeaterAgent) or isinstance(agent, LightAgent) or isinstance(agent, StorageAgent)or isinstance(agent,HeatedFloorAgent)):
                if agent.readyToBuy is True:
                    self.listOfConsumers.append((agent.unique_id,agent.currentDemand))
        self.dictOfConsumers = dict(self.listOfConsumers)
        x = sorted(self.dictOfConsumers.items(), key=operator.itemgetter(1))
        x.reverse()
        self.dictOfConsumers = dict(x)
        print("Consumers {}".format(self.dictOfConsumers))


    def getSellers(self):
        self.numberOfSellers = 0
        self.sellers = []
        self.productions = []
        self.supplyPrice = []
        supplyValue = 0.0
        for agent in self.model.schedule.agents:
            if (isinstance(agent, SolarPanelAgent) or isinstance(agent, StorageAgent) or isinstance(agent,WindEnergyAgent)):
                if agent.readyToSell is True:
                    self.numberOfSellers += 1
                    self.sellers.append(agent.unique_id)
                    self.productions.append(agent.energy)
                    self.supplyPrice.append(agent.price)

                    supplyValue = round(supplyValue + agent.energy, 3)

                    print("Sellers {}".format(agent.unique_id))
        self.totalProduction = sum(self.productions)
        self.historyProductions.append(supplyValue)
        print("Number of sellers {}".format(self.numberOfSellers))
        print("Total production {}".format(self.totalProduction))

    def getBuyres(self):
        self.numberOfBuyers = 0
        self.buyers = []
        self.demands = []
        self.demandPrice = []
        demandValue = 0.0
        for agent in self.model.schedule.agents:
            if (isinstance(agent, HeaterAgent) or isinstance(agent, LightAgent) or isinstance(agent, StorageAgent) or isinstance(agent,HeatedFloorAgent)):
                if agent.readyToBuy is True:
                    self.numberOfBuyers += 1
                    self.buyers.append(agent.unique_id)

                    self.demands.append(agent.currentDemand)
                    demandValue = round(demandValue + agent.currentDemand, 3)

                    self.demandPrice.append(agent.price)
                    print("Buyers {}".format(agent.unique_id))
        self.historyDemands.append(demandValue)
        print("Number of buyers {}".format(self.numberOfBuyers))
        print("Demands {}".format(self.demands))

    def chooseVector(self,vector):
        s = vector
        d = col.defaultdict(list)
        for k, v in s:
            d[k].append(v)
        print(d.items())

    def calculateFitness(self,test_vector):
        fitness = 0
        summedDemand = 0.0
        lst = []
        x = sorted(self.dictOfConsumers.items(), key=operator.itemgetter(1))
        x.reverse()
        for index,elem in enumerate(test_vector):
            if elem > 0:
                lst.append(list(x[index]))
        for elem in lst:
            summedDemand += elem[1]
            summedDemand = round(summedDemand,3)
        fitness = 100 - (self.totalProduction-summedDemand)
        if summedDemand > self.totalProduction:
            fitness = 0.0
        return fitness

    def generatePopulation(self):

        if self.numberOfBuyers > 1:
            popSize = self.numberOfBuyers
        else:
            print("Not enough buyers")
            popSize = self.numberOfBuyers

        vector_list = []
        n, p = 1, 0.5
        d = col.defaultdict(list)

        for i in range(300): #create set of random vectors, calculate fitness of every vector, amount of iterations is essential
            pop = np.random.binomial(n, p, popSize)
            pop = list(pop)
            self.calculateFitness(pop)
            d[self.calculateFitness(pop)].append(pop)
            vector_list.append((pop,self.calculateFitness(pop)))
        print("Population {}".format(vector_list))
        print(d.items())

        ordered_dict = col.OrderedDict(sorted(d.items(), key=lambda t: t[0], reverse=True))
        print("Ordered dict {}".format(ordered_dict))
        sorted_x = sorted(ordered_dict.items(), key=operator.itemgetter(0))
        print("Sorted list {}".format(sorted_x))
        if self.numberOfSellers > 0:
            sorted_x.reverse()
        chosen_elem_list = sorted_x[0]

        print("Best element {}".format(list(chosen_elem_list)[1][0]))
        print("Best fitness {}".format(list(chosen_elem_list)[0]))


        dna = list(chosen_elem_list)[1][0]
        #tournament
        tournament_pool = []
        for i in range(5):
            elem = random.choice(list(d.items()))
            tournament_pool.append(elem)
        print("Tournament pool {}".format(tournament_pool))
        tournament_dict = col.OrderedDict(sorted(tournament_pool, key=lambda t: t[1], reverse=True))
        print(tournament_dict)

        mating_partners = list(tournament_dict.items())

        partners_list = []
        for elem in mating_partners:
            partners_list.append(list(elem)[1][0])
        #get dna for mating
        number_of_partners = 2
        partner = partners_list[0]
        print("Partner {}".format(partner))

        for i in range(300):
            coef = np.random.uniform(0, 1, 1)
            if coef > 0.8:
                dna1 = copy.deepcopy(dna)
                mutated_dna = self.mutate(dna1,self.numberOfBuyers)

                fitness_mutated = self.calculateFitness(mutated_dna)
                fitness_old = self.calculateFitness(dna)
                if fitness_mutated > fitness_old:
                    dna = mutated_dna
            else:
                cross_dna1,cross_dna2 = self.crossover(dna,partner,self.numberOfBuyers)
                fitnes_corss1 = self.calculateFitness(cross_dna1)
                fitnes_corss2 = self.calculateFitness(cross_dna2)
                fitness_old = self.calculateFitness(dna)

                if fitnes_corss1 > fitness_old:
                    dna = cross_dna1

                elif fitnes_corss2 > fitness_old:
                    dna = cross_dna2

        print("Chosen DNA {}".format(dna))

        if self.calculateFitness(dna) <= 0.0:
            self.summedDemands.append(0.0)
            print("Unable to satisfy demand!")
            return 0

        self.decodeList(dna)

    def crossover(self,dna1, dna2, dna_size):
        pos = int(random.random() * dna_size)
        return (dna1[:pos] + dna2[pos:], dna2[:pos] + dna1[pos:])

    def mutate(self,dna,size):
        mutation_chance = 100 #mutation chance
        for index, elem in enumerate(dna):
            if int(random.random() * mutation_chance) == 1:
                if dna[index] == 1:
                    dna[index] = 0
                else:
                    dna[index] = 1
        return dna

    def decodeList(self,dna_variant):
        dnaDemand = 0.0
        self.distributedDemands = []
        buyers_list = list(self.dictOfConsumers.items())
        print(list(buyers_list))
        for index,elem in enumerate(dna_variant):
            if elem > 0:
                print(list(buyers_list[index]))
                self.distributeEnergy(list(buyers_list[index]))
        demand_copy = copy.deepcopy(self.distributedDemands)
        demand_sum = sum(demand_copy)

        self.checkIfConsumersLeft()
        self.summedDemands.append(demand_sum)


    def distributeEnergy(self,data_list)->float:
        summedDemandValue = 0.0
        for agent in self.model.schedule.agents:
            if (isinstance(agent, HeaterAgent) or isinstance(agent, LightAgent) or isinstance(agent, StorageAgent) or isinstance(agent,HeatedFloorAgent)):
                if (agent.unique_id == data_list[0] and agent.readyToBuy is True):
                    print("Agent {}".format(agent.unique_id))
                    print("Current demand {}".format(agent.currentDemand))
                    print("Total energy {}".format(self.totalProduction))
                    self.distributedDemands.append(agent.currentDemand)

                    self.totalProduction = round(self.totalProduction-agent.currentDemand,3)
                    summedDemandValue = round(summedDemandValue + agent.currentDemand,10)

                    agent.currentDemand = 0
                    agent.readyToBuy = False
                    print("Current demand {}".format(agent.currentDemand))
                    print("Total energy {}".format(self.totalProduction))
                    self.dictOfConsumers = self.removeItem(self.dictOfConsumers,agent.unique_id)
        print("Consumers dictionary {}".format(self.dictOfConsumers))
        return summedDemandValue


    def checkIfConsumersLeft(self):
        if self.dictOfConsumers:
            ordered_dict = col.OrderedDict(sorted(self.dictOfConsumers.items(), key=lambda t: t[1],reverse=True))
            print("Consumers left {}".format(ordered_dict.items()))
            for k,v in ordered_dict.items():
                for agent in self.model.schedule.agents:
                    if (isinstance(agent, HeaterAgent) or isinstance(agent, LightAgent) or isinstance(agent,StorageAgent) or isinstance(agent, HeatedFloorAgent)):
                        if (agent.unique_id == k and agent.readyToBuy is True):
                            print("Agent {}".format(agent.unique_id))
                            print("Current demand {}".format(agent.currentDemand))
                            print("Total energy {}".format(self.totalProduction))
                            if self.totalProduction >= agent.currentDemand:
                                self.totalProduction = round(self.totalProduction - agent.currentDemand, 3)

                                agent.currentDemand = 0
                                agent.readyToBuy = False
                                print("Current demand {}".format(agent.currentDemand))
                                print("Total energy {}".format(self.totalProduction))
                                self.dictOfConsumers = self.removeItem(self.dictOfConsumers, agent.unique_id)
                            else:
                                self.buyFromGrid(agent)
                                print("Bought from the grid {}".format(agent.currentDemand))
                                agent.readyToBuy = False
                                print("Current demand {}".format(agent.currentDemand))
                                print("Total energy {}".format(self.totalProduction))
                                self.dictOfConsumers = self.removeItem(self.dictOfConsumers, agent.unique_id)
        else:
            print("All demands are satisfied")


    def checkSurplus(self):
        self.surplus = 0
        if self.totalProduction > 0:
            self.surplus = self.totalProduction

        for agent in self.model.schedule.agents:
            if (isinstance(agent, StorageAgent)):
                print("Stored energy left {}".format(agent.energy))
                energySurplus = agent.addEnergy(self.surplus)
                print("Energy in storage {}".format(agent.energy))
                print("Surplus which can be sold {}".format(energySurplus))

    def buyFromGrid(self,buyer):
        gridPrice = 0
        for agent in self.model.schedule.agents:
            if (isinstance(agent, SmartGridAgent)):
                gridPrice = agent.price
        if buyer.price >= gridPrice:
            buyer.currentDemand = 0

        else:
            print("Bought form Grid {}".format(buyer.currentDemand))
            buyer.currentDemand = 0

    # delete elements from set of buyers
    def removeItem(self,d,key):
        del d[key]
        return d

    def test_func(self):
        print("Control Agent {}".format(self.unique_id))

    def step(self):
        self.test_func()

        self.getSellers()
        self.getBuyres()

        self.getConsumerDict()
        self.generatePopulation()
        self.checkIfConsumersLeft()
        self.checkSurplus()

        self.stepCount += 1
        self.hour +=1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class SolarPanelAgent(Agent):
    def __init__(self,unique_id, model,solarPanel):
        super().__init__(unique_id, model)

        self.solarPanel = solarPanel
        self.energy = 0
        self.readyToSell = True
        self.traided = None
        self.currentDemand = 0

        self.priceHistory = []
        self.quantityHistorySell = []

        self.hour = 0
        self.day = 0
        self.week = 0

    def calculatePrice(self):
        if self.readyToSell:
            self.price = round(random.uniform(1.9,0.3),2)
        else:
            self.price = 0
        print("Price {}".format(self.price))

    def checkSolarEnergy(self):
        self.energy = self.solarPanel.amountOfEnergyGenerated
        print("Amount of Solar energy {}".format(self.solarPanel.amountOfEnergyGenerated))

    def checkStatus(self):
        if self.energy > 0:
            self.readyToSell = True
        else:
            self.readyToSell = False

    def test_func(self):
        print("Seller agent {0}".format(self.unique_id))

    def step(self):
        self.checkSolarEnergy()
        self.checkStatus()
        self.calculatePrice()
        self.traided = False
        print("Ready to Sell {}".format(self.readyToSell))


        self.hour +=1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class WindEnergyAgent(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)
        self.hour = 0
        self.day = 0
        self.week = 0

        self.energy = 0
        self.readyToSell = True
        self.traided = None
        self.currentDemand = 0

    def calculatePrice(self):
        if self.readyToSell:
            self.price = round(random.uniform(1.9,0.3),2)
        else:
            self.price = 0#price for KWh, NOK
        print("Price {}".format(self.price))

    def checkStatus(self):
        if self.energy > 0:
            self.readyToSell = True
        else:
            self.readyToSell = False

    def getWindData(self):
        windList = WIND_DATA
        self.energy = np.random.choice(windList)

    def step(self):
        self.getWindData()
        print("Wind energy {}".format(self.energy))
        self.checkStatus()
        self.calculatePrice()

        self.hour +=1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class HeatedFloorAgent(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)

        self.isInRoom = None
        self.readyToBuy = False

        self.hour = 0
        self.day = 0
        self.week = 0
        self.currentDemand = 0 #0.8 Kwh for average bathroom
        self.price = 0

    def calculatePrice(self):
        self.price = round(random.uniform(1.9,0.3),2)
        print("Price {}".format(self.price))

    def calculateDemand(self):
        if self.isInRoom:
            self.calculatePrice()
            self.currentDemand = 0.8
            self.readyToBuy = True
        else:
            self.currentDemand = 0
            self.readyToBuy = False
        print("Heated floor demand {}".format(self.currentDemand))

    def checkIfisInRoom(self):
        if self.hour >= 0 and self.hour < 7:
            self.isInRoom = np.random.choice(
                [True,False],
                1,
                p=[0.9, 0.1,])[0]
        elif self.hour > 7 and self.hour <= 16:
            if self.day < 5:
                self.isInRoom = np.random.choice(
                    [True, False],
                    1,
                    p=[0.1, 0.9])[0]
            else:
                self.isInRoom = np.random.choice(
                    [True, False],
                    1,
                    p=[0.6, 0.4])[0]
        elif self.hour >= 17 and self.hour <= 21:
            self.isInRoom = np.random.choice(
                [True, False],
                1,
                p=[0.7, 0.3])[0]

        elif self.hour >= 22 and self.hour <= 23:
            self.isInRoom = np.random.choice(
                [True, False],
                1,
                p=[0.9, 0.1])[0]
        print("Person is in the room {}".format(self.isInRoom))

    def test_func(self):
        print("Agent {}".format(self.unique_id))

    def step(self):
        self.test_func()
        self.checkIfisInRoom()
        self.calculateDemand()
        self.hour +=1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class SmartGridAgent(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)

        self.price = 0
        self.tariffCoef = 1

        self.hour = 0
        self.day = 0
        self.week = 0

    def checkTariff(self):
        if self.hour >= 0 and self.hour < 7:
            self.tariffCoef = 0.6

        elif self.hour >= 7 and self.hour <= 10:
            self.tariffCoef = 1.5

        elif self.hour >= 12 and self.hour <= 14:
            self.tariffCoef = 1.9

        elif self.hour >= 18 and self.hour <= 20:
            self.tariffCoef = 1.5

        elif self.hour >= 23:
            self.tariffCoef = 0.5
        else:
            self.tariffCoef = 1
        print("Grid Tariff {}".format(self.tariffCoef))

    def calculatePrice(self):
        self.price = 4*self.tariffCoef
        print("Grid Price {}".format(self.price))

    def test_func(self):
        print("Smart Grid agent {0}".format(self.unique_id))

    def step(self):
        self.checkTariff()
        self.calculatePrice()
        self.hour +=1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class StorageAgent(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)

        self.amperHour = 500
        self.voltage = 24
        self.wCapacity = (self.amperHour*self.voltage)/1000

        self.capacity = self.wCapacity
        self.energy = 12.0
        self.price = 0

        self.readyToSell = True
        self.readyToBuy = False
        self.traided = False

        self.currentDemand = 0
        self.status = None

        self.hour = 0
        self.day = 0
        self.week = 0

        self.priceHistory = []
        self.quantityHistorySell = []
        self.quantityHistoryBuy = []

    def calculateDemand(self):
        if self.status == 'max' or self.status == 'stable':
            self.currentDemand = 0.0
        else:
            self.currentDemand = np.random.choice([(self.capacity/2) - self.energy,(self.capacity-self.energy)])
            self.currentDemand = round(self.currentDemand,3)
        print("Energy demand {}".format(self.currentDemand))


    def checkBatteryCondition(self):
        if self.energy >= 12.0:
            self.energy = 12.0
        if self.energy <= 12.0 and self.energy >= 10.0:
            print("Maximum output, discharge disarable")
            self.status = 'max'
        elif self.energy <=10.0 and self.energy >= 6.0:
            print("Stable with slow discharge")
            self.status = 'stable'
        else:
            self.status = 'unstable'
            print("Unstable state, discharge not desirable, needs charging")

    def calculatePrice(self):
        self.price = round(random.uniform(1.9,0.3),1) #price for KWh, NOK
        print("Price {}".format(self.price))

    def addEnergy(self,energy):
        if (energy + self.energy >= self.capacity):
            print("Possible overcharging")
            surplus = ((self.energy+energy)-self.capacity)
            self.energy += (energy - ((self.energy+energy)-self.capacity))
            self.energy = round(self.energy,3)
            print("Energy level {}".format(self.energy))
        elif energy + self.energy < self.capacity:
            self.energy += energy
            self.energy = round(self.energy, 3)
            surplus = 0
            print("Energy level {}".format(self.energy))
        surplus = round(surplus, 3)
        return surplus

    def getStatus(self):
        print("Status {}".format(self.status))

    def checkStatus(self):
        if self.status == 'max' or self.status == 'stable':
            self.readyToSell = True
            self.readyToBuy = False
        else:
            self.readyToSell = False
            self.readyToBuy = True
        print("Available energy {}".format(self.energy))

    def name_func(self):
        print("Agent {0}".format(self.unique_id))

    def step(self):
        self.name_func()
        self.checkBatteryCondition()
        self.getStatus()
        self.checkStatus()
        self.calculateDemand()
        self.calculatePrice()
        self.traided = False
        print("Ready to Sell {}".format(self.readyToSell))
        print("Ready to Buy {}".format(self.readyToBuy))

        self.hour +=1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class OutdoorLight(object): #level of Solar energy
    def __init__(self):
        self.outdoorLight = 0

class InitAgent(Agent): # weather condition, outdoor temperature
    def __init__(self,unique_id, model,solarPanel,outdoorLight = None,days=7):
        super().__init__(unique_id, model)
        self.weatherCondition = 0

        self.days = days
        self.solarPanel = solarPanel
        self.outLight = outdoorLight
        self.hourCount = 0
        self.outdoorL = 0
        self.outdoorTemp = 0
        self.weatherCoeff = 0

        self.hour = 0
        self.day = 0
        self.week = 0

    def getWeatherCondition(self):
        self.weatherCondition = random.choice(['sunny','partly cloudy','cloudy','rainy'])
        print("Weather is {}".format(self.weatherCondition))
        return self.weatherCondition

    def calculateWeatherCoeff(self):
        if self.weatherCondition == 'sunny':
            self.weatherCoeff = 1.1
        elif self.weatherCondition == 'partly cloudy':
            self.weatherCoeff = 0.8
        elif self.weatherCondition == 'cloudy':
            self.weatherCoeff = 0.2
        elif self.weatherCondition == 'rainy':
            self.weatherCoeff = 0

    def getOutdoorTemp(self):
        self.outdoorTemp = round(np.random.choice(sts.norm.rvs(9, 2, size=24)))
        print("Outdoor temperature {}".format(self.outdoorTemp))

    def calculateSolarEnergy(self):
        amountOfEnergy = 0
        if self.hour >= 0 and self.hour <= 4:
            self.solarPanel.amountOfEnergyGenerated = 0
            amountOfEnergy = 0
            print("Amount of Solar energy {}".format(self.solarPanel.amountOfEnergyGenerated))
        elif self.hour > 4 and self.hour <= 21:
            if self.hour >= 6 and self.hour <=7:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(0.26, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour > 7 and self.hour <= 9:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(0.56, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour > 8 and self.hour <= 10:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(0.97, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour > 10 and self.hour <= 11:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(1.4, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour > 11 and self.hour <= 12:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(2.5, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour > 12 and self.hour <= 13:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(3.68, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour > 13 and self.hour <= 14:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(2.9, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour > 14 and self.hour <= 15:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(1.9, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour > 15 and self.hour <= 16:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(2, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour > 16 and self.hour <= 17:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(1.8, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour > 17 and self.hour <= 18:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(0.8, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour > 18 and self.hour <= 19:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(0.4, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour > 19 and self.hour <= 20:
                amountOfEnergy = abs(round(np.random.choice(sts.norm.rvs(0.1, 1, size=self.days))*self.weatherCoeff,2))
            elif self.hour >= 21:
                amountOfEnergy = 0
            print("Amount of energy Kwh {}".format(amountOfEnergy)) #real data multiplied on weather coefficient
            self.solarPanel.amountOfEnergyGenerated = amountOfEnergy
            print("Amount of Solar energy {}".format(self.solarPanel.amountOfEnergyGenerated))
        else:
            self.solarPanel.amountOfEnergyGenerated = 0
            print("Amount of Solar energy {}".format(self.solarPanel.amountOfEnergyGenerated))

    #amount of lux
    def calculateOutdoorLight(self):
        if self.hour >= 0 and self.hour < 7:
            self.outLight.outdoorLight = 20
        elif self.hour == 7 and self.weatherCondition == 'sunny':
            self.outLight.outdoorLight = 400
        elif self.hour == 7 and self.weatherCondition == 'partly cloudy':
            self.outLight.outdoorLight = 100
        elif self.hour == 7 and self.weatherCondition == 'cloudy':
            self.outLight.outdoorLight = 40
        elif self.hour > 7 and self.hour < 18:
            if self.weatherCondition == 'sunny':
                self.outLight.outdoorLight = 2000
            if self.weatherCondition == 'partly cloudy':
                self.outLight.outdoorLight = 200
            if self.weatherCondition == 'cloudy':
                self.outLight.outdoorLight = 100
        elif self.hour == 18 and self.weatherCondition == 'sunny':
            self.outLight.outdoorLight = 400
        elif self.hour == 18 and self.weatherCondition == 'partly cloudy':
            self.outLight.outdoorLight = 80
        else:
            self.outLight.outdoorLight = 20
        print("Outdoor light {}".format(self.outLight.outdoorLight))


    def step(self):
        print("Hour {}".format(self.hour))
        print("Day {}".format(self.day))
        print("Week {}".format(self.week))

        self.getWeatherCondition()
        self.calculateWeatherCoeff()
        self.calculateSolarEnergy()
        self.calculateOutdoorLight()
        self.getOutdoorTemp()
        self.hourCount += 1
        self.hour +=1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class SolarPanel(object): #level of Solar energy
    def __init__(self,peakPower,sunLevel):
        self.peakPower = peakPower
        self.sunLevel = sunLevel
        self.size = 0
        self.amountOfEnergyGenerated = 0

class HeaterAgent(Agent): #heater agents, heat the rooms, maintain optimal temperature
    def __init__(self, unique_id, model,temperature = 20,roomSize = None):#interact with control agent through the auctions
        super().__init__(unique_id, model) #outdoor temperature, maxTemp, minTemp
        self.energy = 0
        self.minTemp = 20
        self.desiredTemp = 0
        self.readyToBuy = None
        self.price = 0

        self.supply = 0
        self.priceHistory = []
        self.quantityHistoryBuy = []

        #formula coefficients
        self.thermalResistance = 0.1
        self.inertia = 3
        self.outdoorTemp = 16
        self.roomSize = roomSize
        self.power = 1
        self.hour = 0
        self.day = 0
        self.week = 0

        self.isInRoom = True
        self.isInHome = True

        self.turnedOn = True
        self.traided = False

        self.price = 0
        self.demandKwh = 0

        self.gamma = 0.5

        self.tempRange = 0

        self.currentTemp = 0
        self.currentDemand = 0
        self.minDemand = 0
        self.maxDemand = 0

    def calculatePrice(self):
        self.price = round(random.uniform(1.9,0.3),1) #price for KWh, NOK
        print("Price {}".format(self.price))

    def checkStatus(self):
        if self.currentDemand > 0:
            self.readyToBuy = True
        else:
            self.readyToBuy = False
        print("Ready to Buy {}".format(self.readyToBuy))

    def getTempRange(self):
        if self.turnedOn:
            if self.currentTemp > self.minTemp:
                self.tempRange = list(range(self.currentTemp, self.desiredTemp + 1))
            else:
                self.tempRange = list(range(self.minTemp, self.desiredTemp + 1))
            print("Temperature range {}".format(self.tempRange))
        else:
            print("Heater is turned off")

    def checkIfIsIn(self):
        if self.hour >= 0 and self.hour < 7:
            self.isInRoom = np.random.choice([True, False],1,p=[0.9, 0.1, ])[0]
        elif self.hour > 7 and self.hour <= 16:
            if self.day < 5:
                self.isInRoom = np.random.choice([True, False],1,p=[0.1, 0.9])[0]
            else:
                self.isInRoom = np.random.choice([True, False],1,p=[0.6, 0.4])[0]
        elif self.hour >= 17 and self.hour <= 21:
            self.isInRoom = np.random.choice([True, False],1,p=[0.8, 0.2])[0]
        elif self.hour >= 22 and self.hour <= 23:
            self.isInRoom = np.random.choice([True, False], 1, p=[0.9, 0.1])[0]
        print("Person is in the room {}".format(self.isInRoom))

    def getCurrentTemp(self)->int:
        self.previousTemp = self.currentTemp
        self.currentTemp = random.choice(sts.norm.rvs(20,2,size=24))
        self.currentTemp = int(self.currentTemp)
        print("current temperature {}".format(self.currentTemp))
        return self.currentTemp

    def getDesiredTemp(self)->int:
        if self.isInRoom:
            self.desiredTemp = random.choice(list(range(20,31)))#20-30
        else:
            self.desiredTemp = self.minTemp
        print("initial desired temperature {}".format(self.desiredTemp))
        return self.desiredTemp

    def checkTempDifference(self):
        if self.desiredTemp <= self.currentTemp:
            print("no difference or temperature is lower")
            self.turnedOn = False
        else:
            self.turnedOn = True

    def computeDemand(self): #demand
        self.energyKJ = 0
        self.roomSize = 15
        dryAirHeat = 1
        dryAirDencity = 1275
        roomHeight = 2.5
        if self.turnedOn == True:
            self.energyKJ = dryAirHeat*dryAirDencity*self.roomSize*roomHeight*(self.desiredTemp-self.currentTemp)
            self.demandKwh = round((self.energyKJ/1000)*0.00028,3)
            self.currentDemand = self.demandKwh*10
            print("Demand KWh {}".format(self.demandKwh))
        else:
            self.currentDemand = 0

    def name_func(self):
        print("Heater agent {0}".format(self.unique_id))

    def step(self):
        self.name_func()
        self.traided = False

        self.checkIfIsIn()
        self.getCurrentTemp()
        self.getDesiredTemp()

        print("Status {}".format(self.turnedOn))
        self.checkTempDifference()
        print("check temperature difference...")
        print("Status {}".format(self.turnedOn))

        self.getTempRange()
        self.calculatePrice()
        self.computeDemand()
        self.checkStatus()

        print("demand {0}".format(self.currentDemand))
        self.hour +=1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class LightAgent(Agent):
    def __init__(self, unique_id, model, power = 0.075,lumens = 90,area = 15):
        super().__init__(unique_id, model)
        self.energy = 0
        self.power  = power
        self.lumens = lumens #lumens/Watt
        self.utilizationCoeff = 0.6
        self.lightLossFactor = 0.8
        self.area = area
        self.lux = 0
        self.userProfile = 0
        self.readyToBuy = None
        self.traided = None

        self.isInRoom = True

        self.turnedOn = False
        self.bill = 0
        self.price = 0
        self.value = 0

        self.hour = 0
        self.day = 0
        self.week = 0

        self.desiredLight = None
        self.outdoorLight = 0
        self.currentDemand = 0

    def calculatePrice(self):
        self.price = round(random.uniform(1.9,0.3),1) #price for KWh, NOK
        print("Price {}".format(self.price))

    def checkStatus(self):
        if self.currentDemand > 0:
            self.readyToBuy = True
        else:
            self.readyToBuy = False
        print("Ready to Buy {}".format(self.readyToBuy))

    def setUserProfile(self):
        self.userProfile = random.randrange(1,5)
        if self.userProfile == 4:
            self.desiredLight = 1500

        elif self.userProfile == 3:
            self.desiredLight = 500

        elif self.userProfile == 2:
            self.desiredLight = 100

        elif self.userProfile == 1:
            self.desiredLight = 0
            self.turnedOn = False
        print("User profile {}".format(self.userProfile))
        print("Desired Light level {}".format(self.desiredLight))

    def getOutdoorLight(self): #get info from init agent
        for agent in self.model.schedule.agents:
            if (isinstance(agent, InitAgent)):
                self.outdoorLight = agent.outLight.outdoorLight
        print("Calculated outdoor light {}".format(self.outdoorLight))

    def calculateDemand(self):
        if self.turnedOn:
            if self.outdoorLight >= self.desiredLight: #set desired light level according to the user
                powerDemand = (self.desiredLight*self.area)/self.lumens
                self.currentDemand = round(powerDemand / 1000,2)
                print("Desired light {}".format(self.desiredLight))
                print("Power {}".format(powerDemand))

            elif self.outdoorLight < self.desiredLight: #calculate difference for adjusting
                luxDiff = self.desiredLight - self.outdoorLight
                powerDemand = (luxDiff*self.area)/self.lumens
                self.currentDemand = round(powerDemand / 1000,2)
            else:
                self.currentDemand = 0
                print("Too bright")
        else:
            print("no movement")
            self.currentDemand = 0
        print("Light demand {} kW".format(self.currentDemand))

    def getStatus(self):
        if self.isInRoom:
            self.turnedOn = True
        else:
            self.turnedOn = False

    def checkMovement(self):
        if self.hour >= 0 and self.hour < 7:
            self.isInRoom = np.random.choice([True, False],1,p=[0.9, 0.1, ])[0]
        elif self.hour > 7 and self.hour <= 16:
            if self.day < 5:
                self.isInRoom = np.random.choice([True, False],1,p=[0.1, 0.9])[0]
            else:
                self.isInRoom = np.random.choice([True, False],1,p=[0.6, 0.4])[0]
        elif self.hour >= 17 and self.hour <= 21:
            self.isInRoom = np.random.choice([True, False],1,p=[0.8, 0.2])[0]
        elif self.hour >= 22 and self.hour <= 23:
            self.isInRoom = np.random.choice([True, False], 1, p=[0.9, 0.1])[0]
        print("Person is in the room {}".format(self.isInRoom))

    def name_func(self):
        print("Light agent {0}".format(self.unique_id))

    def step(self):
        self.name_func()
        self.traided = False

        self.getOutdoorLight()
        self.setUserProfile()

        self.checkMovement()
        self.getStatus()

        self.calculatePrice()
        self.calculateDemand()
        self.checkStatus()

        self.hour +=1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0


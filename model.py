from mesa import Agent, Model
from schedule import *
from agents import *
from mesa.datacollection import DataCollector#Data collector

class ConceptModel(Model):
    """A model with some number of agents."""

    solarPanel = SolarPanel(3,1) #default = 0
    number_of_active_agents = 4

    verbose = True  # Print-monitoring

    def __init__(self, number_of_active_agents = 2,number_of_light_agents = 1):

        #set params
        self.number_of_active_agents = number_of_active_agents
        self.number_of_light_agents = number_of_light_agents

        self.solarPanel = SolarPanel(3,1)
        self.outdoorLight = OutdoorLight()

        self.schedule = CustomBaseSheduler(self)
        self.datacollector = DataCollector(
            {"Acive agents": lambda m: m.schedule.get_breed_count(HeaterAgent),
             "Light agents": lambda m: m.schedule.get_breed_count(LightAgent)})


        #initialize environment
        initAgent = InitAgent(0,self,solarPanel=self.solarPanel,outdoorLight=self.outdoorLight)
        self.schedule.add(initAgent)

        gridAgent = SmartGridAgent('Grid',self)
        self.schedule.add(gridAgent)

        storageAgent = StorageAgent('Storage',self)
        self.schedule.add(storageAgent)

        solarPanelAgent = SolarPanelAgent('Solar',self,self.solarPanel)
        self.schedule.add(solarPanelAgent)

        windAgent = WindEnergyAgent('Wind',self)
        self.schedule.add(windAgent)

        for i in range(self.number_of_active_agents):
            active_agent = HeaterAgent("Heater agent {}".format(i), self, True)
            self.schedule.add(active_agent)

        #light agents
        for i in range(self.number_of_light_agents):
            light_agent = LightAgent("Light agent {}".format(self.number_of_active_agents+i),self)
            self.schedule.add(light_agent)

        #heated floor
        floorAgent = HeatedFloorAgent('Floor',self)
        self.schedule.add(floorAgent)

        #distribution
        controlAgent = ControlAgent("Control Agent",self)
        self.schedule.add(controlAgent)


    def step(self):
        self.schedule.step()
        """Collect data"""
        self.datacollector.collect(self)
        if self.verbose:
            print([self.schedule.time,
                   self.schedule.get_breed_count(LightAgent),
                   self.schedule.get_breed_count(HeaterAgent)])

    def run_model(self,step_count = 7):

        for i in range(step_count):
            print("Step {}".format(i))
            for j in range(24):
                self.step()
import json
import random

import torch

from . import Logging as LOG

class EpisodeLoader():
    def __init__(self, DataSet, DataSetParameters):
        self.Samples                           : list[str] = DataSetParameters["Samples"]
        self.Classes                           : list[str] = DataSetParameters["Classes"]
        self.Images                            : dict      = DataSet

        self.NumberOfSupportSamplesPerClass    : int = 1
        self.NumberOfQuerySamplesPerClass      : int = 1
        self.NumberOfValidationSamplesPerClass : int = 1

        self.LabelMap    : dict = {}

        self.Initialize()

    def Initialize(self):
        self.EpisodicClasses = [f"{Sample}_{Class}" for Sample in self.Samples for Class in self.Classes]

    def GenerateLabelMap(self) -> None:
        self.LabelMap = {}
        for Index, Class in enumerate(self.EpisodicClasses):
            self.LabelMap[Class] = Index
        with open("Outputs/Labelmap.json", "w") as f:
            json.dump(self.LabelMap, f)

    def GetRandomEpisode(self, LogFile, Controls, Mode : str = "Training") -> dict:
        '''
        Creates a random episode out of the provided data set
        following Algorithm 1 from Snell et al. (2017)
        doi: doi.org/10.48550/arXiv.1703.05175
        '''

        Episode = {}


        '''
        Get maximum samples for each class -> generate random indices based on this
            make sure that all images CAN be included (none should be out of range)
        Slice indices
        generate Episode
        '''


        return Episode

        '''
        self.RequestedNC = Controls.NumberOfClassesPerEpisode
        self.GetEpisodeData()
        
        # Assumes that the data set is balanced, meaning the same classes and modes for each input sample.

        if not self.Images:
            raise Exception("No data set available. Load data set before creating episodes.")

        SelectedClasses = random.sample(self.EpisodicClasses, self.NumberOfClassesPerEpisode)

        AllIndices   = torch.randperm(self.MaximumSamplesPerClass)
        SupportRange = self.NumberOfSupportSamplesPerClass
        QueryRange   = self.NumberOfSupportSamplesPerClass + self.NumberOfQuerySamplesPerClass
        SupportIndices    : list[int] = AllIndices[             : SupportRange].tolist()
        QueryIndices      : list[int] = AllIndices[SupportRange : QueryRange  ].tolist()
        ValidationIndices : list[int] = AllIndices[QueryRange   :             ].tolist()

        Episode = {}

        ClassTensors = []
        for Class in SelectedClasses:
            PreprocessedSupportTensors = []
            for Index in SupportIndices:
                PreprocessedSupportTensors.append(self.Images[Class.split("_")[0]][Class.split("_")[1]][Mode]["DinoFeatures"][Index])
            ClassTensors.append(torch.stack(PreprocessedSupportTensors))
        Episode["SupportTensor"] = torch.stack(ClassTensors)
        
        ClassTensors = []
        for Class in SelectedClasses:
            PreprocessedQueryTensors   = []
            for Index in QueryIndices:
                PreprocessedQueryTensors.append(self.Images[Class.split("_")[0]][Class.split("_")[1]][Mode]["DinoFeatures"][Index])
            ClassTensors.append(torch.stack(PreprocessedQueryTensors))
        Episode["QueryTensor"] = torch.stack(ClassTensors)
        
        ClassTensors = []
        for Class in SelectedClasses:
            PreprocessedValidationTensors   = []
            for Index in ValidationIndices:
                PreprocessedValidationTensors.append(self.Images[Class.split("_")[0]][Class.split("_")[1]][Mode]["DinoFeatures"][Index])
            ClassTensors.append(torch.stack(PreprocessedValidationTensors))
        Episode["ValidationTensor"] = torch.stack(ClassTensors)

        if Controls.WriteDetailedDebugInfo:
            LOG.TrainingEpisode(LogFile, Controls, SelectedClasses, SupportIndices, QueryIndices, ValidationIndices, Episode)

        return Episode

    def GetEpisodeData(self, RequestedNC: int = 2, RequestedNS: int = 5, RequestedNQ: int = 1):
        
        
        if self.NumberOfClassesPerEpisode > len(self.EpisodicClasses):
            self.NumberOfClassesPerEpisode = len(self.EpisodicClasses)
        
        # Calculate the continuous solution
        AvailableSamples = int(0.75 * self.NumberOfClassesPerEpisode)
        SupportToQueryRatio = RequestedNS/RequestedNQ
    
        NQ_Ideal = AvailableSamples / (SupportToQueryRatio + 1)
        NS_Ideal = SupportToQueryRatio * NQ_Ideal

        # Search integer solutions near continuous solution
        BestRatioError = float("inf")
        SearchRadius   = 3
        SolutionFound  = False
        for NS in range(max(1, int(NS_Ideal) - SearchRadius), int(NS_Ideal) + SearchRadius):
            for NQ in range(max(1, int(NQ_Ideal) - SearchRadius), int(NQ_Ideal) + SearchRadius):
                if NS + NQ > AvailableSamples:
                    continue
                RatioError = abs(NS/NQ - SupportToQueryRatio)
                if RatioError < BestRatioError:
                    BestRatioError     = RatioError
                    self.NumberOfSupportSamplesPerClass = NS
                    self.NumberOfQuerySamplesPerClass   = NQ
                    SolutionFound = True
        if not SolutionFound:
            self.NumberOfSupportSamplesPerClass = 1
            self.NumberOfQuerySamplesPerClass   = 1
        
        self.NumberOfValidationSamplesPerClass = AvailableSamples - self.NumberOfSupportSamplesPerClass - self.NumberOfQuerySamplesPerClass
        '''
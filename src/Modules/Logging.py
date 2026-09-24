import os
import h5py
import numpy as np
import torch

class NoteFile():
    def __init__(self, FilePath: str):
        self.FilePath = FilePath
        os.makedirs(os.path.dirname(FilePath), exist_ok=True)
        self.File = open(FilePath, "w")

    def W(self, Text, NewLine: bool = True):
        self.File.write(str(Text) + "\n" if NewLine else str(Text))
        self.File.flush()

    def Close(self):
        self.File.close()

    def __del__(self):
        if not self.File.closed:
            self.File.close()

def LogDict(Log, Dictionary: dict, Indent: int = 0):
    for Key, Value in Dictionary.items():
        Prefix = "  " * Indent
        if isinstance(Value, dict):
            Log.W(f"{Prefix}{Key}:")
            LogDict(Log, Value, Indent + 1)
        elif isinstance(Value, torch.Tensor) and len(Value) > 0:
            Log.W(f"{Prefix}{Key}: Tensor {list(Value.shape)}")
        elif isinstance(Value, (list, np.ndarray)) and len(Value) > 0:
            Log.W(f"{Prefix}{Key} ({len(Value)} entries): {str(Value[:3]).rstrip(']')} ...]")
        else:
            Log.W(f"{Prefix}{Key}: {str(Value)[:50]}")

def LogHdf5Write(Log, Controls, HdfFile):
    Log.W(f"\nHDF5 File created as Outputs/{Controls.DataSet}_DINOv2Features.h5")
    Log.W(f"Meta data:")
    for Key, Value in HdfFile["MetaData"].attrs.items():
        Log.W(f"    {Key}: {Value}")
    def PrintLeaf(Name, Object):
        if isinstance(Object, h5py.Dataset):
            Log.W(f"    {Name}")
    Log.W(f"HDF5 paths:")
    HdfFile.visititems(PrintLeaf)

def ApplyDinoNetwork_FirstInstance(Log, MaxBatchSize, BatchSize, NumberOfBatches, DetailedLog, Images):
    Log.W(f"Maximum Batch Size: {MaxBatchSize}, Calculated Batch Size: {BatchSize}, Resulting Number of Batches: {NumberOfBatches}")
        
    if DetailedLog:
        for Index, _ in enumerate(Images["Images"]):
            Log.W(f"    {Index:>5} - {Images["Names"][Index]}: {Images["Images"][Index]}")
    
        Log.W("    Batching:")

def ApplyDinoNetwork_SecondInstance(Log, DetailedLog, BatchedImages, BatchFeatures, BatchedNames):
    if DetailedLog:
        for Index, _ in enumerate(BatchedImages):
            Vector = BatchFeatures[Index]
            Log.W(f"        {Index:>5} {BatchedNames[Index]:<30} [1 x {len(Vector):>5}] [{", ".join(f"{Entry: 7.4f}" for Entry in Vector[:5])}, ..., {", ".join(f"{Entry: 7.4f}" for Entry in Vector[-5:])}]")

def ApplyDinoNetwork_ThirdInstance(DetailedLog, Log, FeatureTensor, Images):
    if DetailedLog:
        Log.W("    Extracted Features:")
        Log.W(f"    Index   Name                           Feature Vector                                                                                  PIL Image")
        for Index in range(FeatureTensor.shape[0]):
            Vector = FeatureTensor[Index]
            Log.W(f"    {Index:>5} - {Images["Names"][Index]}: [{", ".join(f"{Entry: 7.4f}" for Entry in Vector[:5])}, ..., {", ".join(f"{Entry: 7.4f}" for Entry in Vector[-5:])}] {Images["Images"][Index]}")

def GenerateEncoder(Log, FeatureDimension, EmbeddingDimension, NumberOfHiddenLayers, Gamma, Indices, Dimensions):
    Log.W(f"Reducing from {FeatureDimension} to {EmbeddingDimension} Dimensions with {NumberOfHiddenLayers} hidden layers - Gamma = {Gamma}")
    Log.W("Indices   : [", NewLine=False)
    for Index in Indices:
        Log.W(f"{Index:>4}", NewLine=False)
    Log.W("]\nDimensions: [", NewLine=False)
    for Dimension in Dimensions:
        Log.W(f"{Dimension:>4}", NewLine=False)
    Log.W("]")

def Training(Controls, EpisodeLoader, Log):
    Log.W("\nStart main training loop\n==========================")
    Log.W("*-------------------------------------------------------------------------*")
    Log.W("| Parameter                           | Variable | Value   | Value        |")
    Log.W("|                                     |          | (Input) | (Calculated) |")
    Log.W("|-------------------------------------|----------|---------|--------------|")
    Log.W(f"| Number of Episodes                  |          | {Controls.NumberOfEpisodes:<4}    |              |")
    Log.W(f"| Number of Classes per Episode       | N_C      | {Controls.NumberOfClassesPerEpisode:<4}    | {EpisodeLoader.NumberOfClassesPerEpisode:<4}         |")
    Log.W(f"| Number of Support Samples per Class | N_S      | {Controls.NumberOfSupportSamplesPerClass:<4}    | {EpisodeLoader.NumberOfSupportSamplesPerClass:<4}         |")
    Log.W(f"| Number of Query Samples per Class   | N_Q      | {Controls.NumberOfQuerySamplesPerClass:<4}    | {EpisodeLoader.NumberOfQuerySamplesPerClass:<4}         |")
    Log.W("*-------------------------------------------------------------------------*")

def WritelabelMap(Log, LabelMap):
    Log.W("\n*------------------------------*")
    Log.W("| Class                | Label |")
    Log.W("|----------------------|-------|")
    for Class in LabelMap.keys():
        Log.W(f"| {Class:<20} | {LabelMap[Class]:<5} |")
    Log.W("*------------------------------*")

def TrainingEpisode(Log, Controls, SelectedClasses, SupportIndices, QueryIndices, ValidationIndices, Episode):
    Log.W(f"  Selected classes   : {SelectedClasses}")
    Log.W(f"  Support indices    : {SupportIndices}")
    Log.W(f"  Query indices      : {QueryIndices}")
    Log.W(f"  Validation indices : {ValidationIndices}")
    
    Log.W(f"\n  Tensor shapes:")
    Log.W(f"  Support tensor   : [{Episode["SupportTensor"].size(0):>3}, {Episode["SupportTensor"].size(1):>3}, {Episode["SupportTensor"].size(2):>4}]")
    Log.W(f"  Query tensor     : [{Episode["QueryTensor"].size(0):>3}, {Episode["QueryTensor"].size(1):>3}, {Episode["QueryTensor"].size(2):>4}]")
    Log.W(f"  Validation tensor: [{Episode["ValidationTensor"].size(0):>3}, {Episode["ValidationTensor"].size(1):>3}, {Episode["ValidationTensor"].size(2):>4}]")

def PrintTrainingProcess(Log, Controls, RunningLoss, RunningAccuracy):
    Log.W("\nResults of the training process:")
    Log.W("----------------------------------")

    Log.W("*--------------------------------*")
    Log.W("| Episode | Loss    | Validation |")
    Log.W("| Index   |         | Accuracy   |")
    Log.W("|---------|---------|------------|")
    for Index, _ in enumerate(RunningAccuracy):
        Log.W(f"| {Index+1:<7} | {round(RunningLoss[Index],5):<7} | {round(100*RunningAccuracy[Index],7):<10} |")
    Log.W("*--------------------------------*")

    with open(f"{Controls.OutputPath}TrainingProcess.csv", "w") as TrainingProcessCsv:
        TrainingProcessCsv.write("Episode Index,         Loss, Validation Accuracy\n")
        for Index, _ in enumerate(RunningAccuracy):
            TrainingProcessCsv.write(f"{Index+1:>13}, {round(RunningLoss[Index],10):12.8f}, {round(100*RunningAccuracy[Index],17):19.15f}\n")
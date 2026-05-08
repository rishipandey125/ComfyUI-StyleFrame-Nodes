from .nodes import *


#  Map all your custom nodes classes with the names that will be displayed in the UI.

#nodes I will need 

#Pack Frames 
#Unpack Frames

NODE_CLASS_MAPPINGS = {
    "Threshold Image": ThresholdImage,
    "Resize Frame": ResizeFrame,
    "Pad Batch to kn+1": PadBatchToNPlus1,
    "Pad Batch to 4n+1": PadBatchToNPlus1,
    "Trim Padded Batch": TrimPaddedBatch,
    "Get Image Dimensions": GetImageDimensions,
    "Slot Frame": SlotFrame,
    "Batch Keyframes": BatchKeyframes,
    "Select Image From Batch": SelectImageFromBatch,
    "Canny Edge": CannyEdge,
    "Load Image Folder": LoadImageFolder,
    "Load Images": LoadImages,
    "Load Images From List": LoadImagesFromList,
    "Save Image Folder": SaveImageFolder,
    "String To Float List": StringToFloatList,
    "Image Grayscale": ImageGrayscale,
    "Create Empty Frames": CreateEmptyFrames,
    "Split RGB Channels": SplitRGBChannels,
    "Combine RGB Channels": CombineRGBChannels,
    "LTXVMultiKeyframeGuide": LTXVMultiKeyframeGuide,
    "LTXVMultiLatentGuide": LTXVMultiLatentGuide,
    "Crop Guide Frames": CropGuideFrames,
    "Repeat Frames": RepeatFrames,
    "Resample Frames": ResampleFrames,
}


__all__ = ['NODE_CLASS_MAPPINGS']

from .nodes import *


#  Map all your custom nodes classes with the names that will be displayed in the UI.

#nodes I will need 

#Pack Frames 
#Unpack Frames

NODE_CLASS_MAPPINGS = {
    "Threshold Image": ThresholdImage,
    "Resize Frame": ResizeFrame,
    "Pad Batch to 4n+1": PadBatchTo4nPlus1,
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
}


__all__ = ['NODE_CLASS_MAPPINGS']

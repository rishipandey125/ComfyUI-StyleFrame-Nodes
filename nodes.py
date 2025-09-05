#  Package Modules
import os
from typing import Union, BinaryIO, Dict, List, Tuple, Optional
import time

#  ComfyUI Modules
import folder_paths
from comfy.utils import ProgressBar, common_upscale
import node_helpers

import cv2
import numpy as np
import math 
import torch 

from PIL import Image

# Collection of helper nodes for building comfy workflows that power styleframe ai

def convert_tensor_to_numpy(tensor):
    """ Convert tensor to numpy array and scale it properly for image processing. """
    return (tensor.detach().cpu().numpy() * 255).astype(np.uint8)

def convert_numpy_to_tensor(numpy_image):
    """ Convert processed numpy image back to tensor and normalize it. """
    return torch.from_numpy(numpy_image).float() / 255


class BatchKeyframes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], { "default": "lanczos" }),
            }, "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image batch"

    def execute(self, image_1, method, image_2=None, image_3=None, image_4=None, image_5=None):
        out = image_1

        if image_2 is not None:
            if image_1.shape[1:] != image_2.shape[1:]:
                image_2 = comfy.utils.common_upscale(image_2.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((image_1, image_2), dim=0)
        if image_3 is not None:
            if image_1.shape[1:] != image_3.shape[1:]:
                image_3 = comfy.utils.common_upscale(image_3.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_3), dim=0)
        if image_4 is not None:
            if image_1.shape[1:] != image_4.shape[1:]:
                image_4 = comfy.utils.common_upscale(image_4.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_4), dim=0)
        if image_5 is not None:
            if image_1.shape[1:] != image_5.shape[1:]:
                image_5 = comfy.utils.common_upscale(image_5.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_5), dim=0)

        return (out,)


class ResizeFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_width": ("INT", {"default": 4, "min": 1, "tooltip": "Input frame width"}),
                "frame_height": ("INT", {"default": 4, "min": 1, "tooltip": "Input frame height"}),
                "target_p": ("INT", {"default": 1080, "min": 16, "tooltip": "Target short side in pixels (e.g., 720, 1080)"}),
                "allow_upscale": ("BOOLEAN", {"default": False, "tooltip": "If enabled, scale up to reach target p"}),
                "rounding": ((["nearest", "floor", "ceil"]), {"default": "nearest", "tooltip": "How to choose the 16-multiple step near target p"}),
            },
        }

    RETURN_TYPES = ("INT","INT")
    RETURN_NAMES = ("width","height")
    FUNCTION = "resize_frame"
    CATEGORY = "Image Processing"

    def resize_frame(self, frame_width, frame_height, target_p, allow_upscale, rounding):
        # Reduce aspect ratio to the simplest integer ratio a:b
        gcd = math.gcd(frame_width, frame_height)
        a = frame_width // gcd
        b = frame_height // gcd

        # Step size for the shorter side in pixels when preserving AR and using multiples of 16
        short_unit = 16 * min(a, b)
        width_unit = 16 * a
        height_unit = 16 * b

        # Helper to clamp at least 1
        def at_least_one(x):
            return max(1, x)

        def quantize_to_16(value, mode):
            if mode == "floor":
                return (int(value) // 16) * 16
            elif mode == "ceil":
                return ((int(value) + 15) // 16) * 16
            else:  # nearest
                lower = (int(value) // 16) * 16
                upper = lower + 16
                if abs(value - lower) <= abs(upper - value):
                    return lower
                else:
                    return upper

        # Determine t that best matches target_p for the shorter side
        # Compute target t by rounding rule
        ideal_t = target_p / short_unit
        if rounding == "floor":
            t_candidate = math.floor(ideal_t)
        elif rounding == "ceil":
            t_candidate = math.ceil(ideal_t)
        else:  # "nearest"
            t_floor = math.floor(ideal_t)
            t_ceil = math.ceil(ideal_t)
            # Prefer the one closer to ideal; on tie prefer down to avoid surprise upscale
            if abs(ideal_t - t_floor) <= abs(t_ceil - ideal_t):
                t_candidate = t_floor
            else:
                t_candidate = t_ceil

        # Ensure minimum t of 1 to keep 16-multiple validity
        t_candidate = at_least_one(t_candidate)

        # Respect downscale-only by clamping to not exceed original dims when allow_upscale is False
        if not allow_upscale:
            # max t that does not exceed original dimensions
            t_max_by_original_w = frame_width // width_unit if width_unit > 0 else 0
            t_max_by_original_h = frame_height // height_unit if height_unit > 0 else 0
            t_max_by_original = min(t_max_by_original_w, t_max_by_original_h)
            if t_max_by_original <= 0:
                # If original is smaller than the minimal 16-multiple, fallback to t=1 (minimum valid size)
                t = 1
            else:
                # If nearest picks a value above original, clamp down
                t = min(t_candidate, t_max_by_original)
        else:
            t = t_candidate

        # Compute final dimensions preserving exact AR and multiples of 16
        new_width = width_unit * t
        new_height = height_unit * t

        # Fallback: if we couldn't reduce size under exact-AR constraint (common when gcd is 16)
        # and downscale-only is requested, approximate AR to hit target_p on the shorter side.
        # Condition: target short side smaller than the exact-AR minimum short_unit*t (here t>=1)
        exact_short = min(new_width, new_height)
        target_p_down = max(16, (target_p // 16) * 16)
        if not allow_upscale and target_p_down < exact_short:
            # Scale factor based on the shorter side
            shorter_side = min(frame_width, frame_height)
            scale = target_p / max(1, shorter_side)
            scale = min(1.0, scale)  # downscale-only

            scaled_w = frame_width * scale
            scaled_h = frame_height * scale

            q_w = quantize_to_16(scaled_w, rounding)
            q_h = quantize_to_16(scaled_h, rounding)

            # Ensure shorter side does not exceed target_p when downscaling
            if min(q_w, q_h) > target_p_down:
                if q_w <= q_h:
                    q_w = min(q_w, target_p_down)
                    q_w = (q_w // 16) * 16
                else:
                    q_h = min(q_h, target_p_down)
                    q_h = (q_h // 16) * 16

            # Ensure we did not exceed original dimensions in downscale-only mode
            q_w = min(q_w, frame_width)
            q_h = min(q_h, frame_height)

            # Apply minimum bound
            new_width = max(16, q_w)
            new_height = max(16, q_h)

        # Ensure we don't end up with zero or negative dimensions
        new_width = max(16, new_width)
        new_height = max(16, new_height)

        return (int(new_width), int(new_height))


class PadBatchTo4nPlus1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to be padded to 4n+1 size"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("padded_images", "original_count", "total_count")
    FUNCTION = "pad_batch"
    CATEGORY = "Image Processing"

    def pad_batch(self, images):
        # Get the current batch size
        current_size = images.shape[0]
        
        # Calculate the next 4n+1 size
        n = math.ceil((current_size - 1) / 4)
        target_size = 4 * n + 1
        
        # If we're already at a 4n+1 size, return as is
        if current_size == target_size:
            return (images, current_size, target_size)
            
        # Calculate how many frames we need to pad
        padding_size = target_size - current_size
        
        # Get the last frame to repeat
        last_frame = images[-1]
        
        # Create the padding frames by repeating the last frame
        padding_frames = last_frame.unsqueeze(0).repeat(padding_size, 1, 1, 1)
        
        # Concatenate the original frames with the padding frames
        padded_images = torch.cat([images, padding_frames], dim=0)
        
        return (padded_images, current_size, target_size)

class TrimPaddedBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Padded images to be trimmed"}),
                "original_count": ("INT", {"default": 1, "min": 1, "tooltip": "Original number of frames before padding"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("trimmed_images",)
    FUNCTION = "trim_batch"
    CATEGORY = "Image Processing"

    def trim_batch(self, images, original_count):
        # Ensure original_count is not larger than the current batch size
        original_count = min(original_count, images.shape[0])
        
        # Trim the batch to the original size
        trimmed_images = images[:original_count]
        
        return (trimmed_images,)

class GetImageDimensions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to get dimensions from"}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_dimensions"
    CATEGORY = "Image Processing"

    def get_dimensions(self, image):
        # Get dimensions from the image tensor
        # Image tensor shape is [batch, height, width, channels]
        height, width = image.shape[1:3]
        return (width, height)


#Create a node that accepts keyframe indices and keyframes + control path (all gray if one is not provided) and returns the correct sequence + mask capability
class SlotFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "empty_frame_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "White level of empty frame to use"}),
            "slot_image": ("IMAGE", {"tooltip": "Image to place in the specified slot"}),
            "slot_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Index where to place the slot image"}),
            },
            "optional": {
                "control_images": ("IMAGE",),
                "inpaint_mask": ("MASK", {"tooltip": "Inpaint mask to use for the empty frames"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "process"
    CATEGORY = "Image Processing"
    DESCRIPTION = "Places a single image at a specified index in a sequence of frames"

    def process(self, num_frames, empty_frame_level, slot_image, slot_index, control_images=None, inpaint_mask=None):
        B, H, W, C = slot_image.shape
        device = slot_image.device

        masks = torch.ones((num_frames, H, W), device=device)

        if control_images is not None:
            control_images = common_upscale(control_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
        
        # Create empty frames
        if control_images is None:
            empty_frames = torch.ones((num_frames, H, W, 3), device=device) * empty_frame_level
        else:
            empty_frames = control_images[:num_frames]
        
        # Ensure slot_index is within bounds
        slot_index = min(slot_index, num_frames - 1)
        
        # Replace the frame at slot_index with slot_image
        empty_frames[slot_index:slot_index + slot_image.shape[0]] = slot_image
        
        # Create mask for the slot image
        masks[slot_index:slot_index + slot_image.shape[0]] = 0

        if inpaint_mask is not None:
            inpaint_mask = common_upscale(inpaint_mask.unsqueeze(1), W, H, "nearest-exact", "disabled").squeeze(1).to(device)
            if inpaint_mask.shape[0] > num_frames:
                inpaint_mask = inpaint_mask[:num_frames]
            elif inpaint_mask.shape[0] < num_frames:
                inpaint_mask = inpaint_mask.repeat(num_frames // inpaint_mask.shape[0] + 1, 1, 1)[:num_frames]

            empty_mask = torch.ones_like(masks, device=device)
            masks = inpaint_mask * empty_mask
    
        return (empty_frames.cpu().float(), masks.cpu().float())

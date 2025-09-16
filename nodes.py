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
import json

from PIL import Image

# Collection of helper nodes for building comfy workflows that power styleframe ai

def convert_tensor_to_numpy(tensor):
    """ Convert tensor to numpy array and scale it properly for image processing. """
    return (tensor.detach().cpu().numpy() * 255).astype(np.uint8)

def convert_numpy_to_tensor(numpy_image):
    """ Convert processed numpy image back to tensor and normalize it. """
    return torch.from_numpy(numpy_image).float() / 255


def run_canny_on_pil(pil_image: Image.Image, low_threshold: int, high_threshold: int) -> Image.Image:
    # OpenCV Canny implementation only (no controlnet_aux dependency)
    rgb = pil_image.convert("RGB")
    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return Image.fromarray(edges).convert("RGB")


class LoadImageFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": "", "tooltip": "Folder containing input images"}),
                "extensions": ("STRING", {"default": "png,jpg,jpeg,webp", "tooltip": "Comma-separated extensions or 'all'"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "filenames", "sizes")
    FUNCTION = "load_folder"
    CATEGORY = "Image Processing/IO"

    def _list_images(self, folder_path: str, extensions: list) -> list:
        files = []
        try:
            for name in os.listdir(folder_path):
                p = os.path.join(folder_path, name)
                if not os.path.isfile(p):
                    continue
                ext = os.path.splitext(name)[1].lower().lstrip(".")
                if len(extensions) == 0 or ext in extensions:
                    files.append(p)
        except FileNotFoundError:
            return []
        files.sort()
        return files

    def load_folder(self, input_folder: str, extensions: str):
        input_folder = os.path.expandvars(os.path.expanduser(input_folder)).strip().strip('"').strip("'")
        ext_list = [e.strip().lower().lstrip('.') for e in extensions.split(',') if e.strip() != ""]
        if len(ext_list) == 0 or "*" in ext_list or "all" in ext_list:
            ext_list = ["png", "jpg", "jpeg", "webp"]

        paths = self._list_images(input_folder, ext_list)
        if len(paths) == 0:
            # Return empty batch
            empty = torch.zeros((0, 1, 1, 3), dtype=torch.float32)
            return (empty, json.dumps([]), json.dumps([]))

        images = []
        names = []
        sizes = []
        max_w = 0
        max_h = 0
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                h, w = arr.shape[0], arr.shape[1]
                sizes.append([w, h])
                if w > max_w:
                    max_w = w
                if h > max_h:
                    max_h = h
                images.append(arr)
                names.append(os.path.basename(p))
            except Exception:
                continue

        if len(images) == 0:
            empty = torch.zeros((0, 1, 1, 3), dtype=torch.float32)
            return (empty, json.dumps([]), json.dumps([]))

        # Pad images to (max_h, max_w)
        padded_tensors = []
        for arr in images:
            h, w = arr.shape[0], arr.shape[1]
            if h == max_h and w == max_w:
                padded = arr
            else:
                padded = np.zeros((max_h, max_w, 3), dtype=np.float32)
                padded[:h, :w, :] = arr
            padded_tensors.append(torch.from_numpy(padded))

        batch = torch.stack(padded_tensors, dim=0)
        return (batch, json.dumps(names), json.dumps(sizes))


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
            "slot_indices": ("STRING", {"default": "0", "multiline": False, "tooltip": "Comma-separated indices (e.g., 0,3,56) where to place the slot images"}),
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
    DESCRIPTION = "Places one or more images at specified indices in a sequence of frames"

    def process(self, num_frames, empty_frame_level, slot_image, slot_indices, control_images=None, inpaint_mask=None):
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

        # Parse indices from string
        indices_list = []
        if isinstance(slot_indices, str):
            # Support commas or semicolons as separators
            for token in slot_indices.replace(";", ",").split(","):
                t = token.strip()
                if t == "":
                    continue
                try:
                    idx = int(t)
                except ValueError:
                    continue
                # Clamp to valid range
                if num_frames > 0:
                    idx = max(0, min(num_frames - 1, idx))
                indices_list.append(idx)

        if len(indices_list) == 0:
            indices_list = [0]

        # Place images at specified indices
        if B == len(indices_list):
            for i, idx in enumerate(indices_list):
                empty_frames[idx] = slot_image[i]
                masks[idx] = 0
        elif B == 1:
            for idx in indices_list:
                empty_frames[idx] = slot_image[0]
                masks[idx] = 0
        else:
            # Map by position up to the shortest length
            limit = min(B, len(indices_list))
            for i in range(limit):
                idx = indices_list[i]
                empty_frames[idx] = slot_image[i]
                masks[idx] = 0

        if inpaint_mask is not None:
            inpaint_mask = common_upscale(inpaint_mask.unsqueeze(1), W, H, "nearest-exact", "disabled").squeeze(1).to(device)
            if inpaint_mask.shape[0] > num_frames:
                inpaint_mask = inpaint_mask[:num_frames]
            elif inpaint_mask.shape[0] < num_frames:
                inpaint_mask = inpaint_mask.repeat(num_frames // inpaint_mask.shape[0] + 1, 1, 1)[:num_frames]

            # Combine existing mask (zeros at placed indices) with provided inpaint mask
            masks = masks * inpaint_mask
    
        return (empty_frames.cpu().float(), masks.cpu().float())


class SelectImageFromBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input image batch"}),
                "index": ("INT", {"default": -1, "min": -999999, "tooltip": "Index in batch (supports negative like -1 for last)"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "select_index"
    CATEGORY = "Image Processing"

    def select_index(self, images, index):
        # images: [batch, height, width, channels]
        batch_size = images.shape[0]
        if batch_size == 0:
            # Nothing to select; return as-is
            return (images,)

        # Normalize negative indices
        if index < 0:
            index = batch_size + index

        # Clamp to valid range
        index = max(0, min(batch_size - 1, int(index)))

        # Keep batch dimension
        selected = images[index:index+1]
        return (selected,)


class CannyEdge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input image or batch"}),
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edges",)
    FUNCTION = "execute"
    CATEGORY = "Image Processing/Filters"

    def execute(self, images, low_threshold: int, high_threshold: int):
        # images [B,H,W,C] in 0..1 float
        low_threshold = int(max(0, min(255, low_threshold)))
        high_threshold = int(max(0, min(255, high_threshold)))
        if high_threshold < low_threshold:
            low_threshold, high_threshold = high_threshold, low_threshold

        batch, height, width, channels = images.shape
        device = images.device
        output = []
        for i in range(batch):
            pil_in = Image.fromarray((images[i].cpu().numpy() * 255).astype(np.uint8))
            pil_out = run_canny_on_pil(pil_in, low_threshold, high_threshold)
            arr = np.array(pil_out).astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            # Ensure original resolution is preserved explicitly
            if arr.shape[0] != height or arr.shape[1] != width:
                arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST)
            output.append(torch.from_numpy(arr))
        result = torch.stack(output, dim=0).to(device)
        return (result,)



class SaveImageFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Image batch to save"}),
                "output_folder": ("STRING", {"default": "", "tooltip": "Folder to save images"}),
                "format": ("STRING", {"default": "png", "tooltip": "png|jpg|webp"}),
            },
            "optional": {
                "filenames": ("STRING", {"default": "[]", "tooltip": "Optional JSON array of filenames to use directly"}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("saved_count", "output_folder")
    FUNCTION = "save_folder"
    CATEGORY = "Image Processing/IO"

    def save_folder(self, images, output_folder: str, format: str, filenames: str = "[]"):
        output_folder = os.path.expandvars(os.path.expanduser(output_folder)).strip().strip('"').strip("'")
        os.makedirs(output_folder, exist_ok=True)

        try:
            name_list = json.loads(filenames)
            if not isinstance(name_list, list):
                name_list = []
        except Exception:
            name_list = []

        ext = format.lower().strip('.').strip()
        if ext not in ["png", "jpg", "jpeg", "webp"]:
            ext = "png"

        count = images.shape[0]
        saved = 0
        pbar = ProgressBar(count)
        for i in range(count):
            try:
                if i < len(name_list) and isinstance(name_list[i], str) and name_list[i].strip() != "":
                    candidate = os.path.basename(name_list[i].strip())
                    base, ext_existing = os.path.splitext(candidate)
                    if ext_existing == "":
                        out_filename = f"{candidate}.{ext}"
                    else:
                        out_filename = candidate  # keep provided extension
                else:
                    out_filename = f"output_{i}.{ext}"

                out_path = os.path.join(output_folder, out_filename)

                arr = (images[i].cpu().numpy() * 255).clip(0,255).astype(np.uint8)
                pil = Image.fromarray(arr)
                params = {}
                out_ext = os.path.splitext(out_path)[1].lower()
                if out_ext in [".jpg", ".jpeg"]:
                    if pil.mode != "RGB":
                        pil = pil.convert("RGB")
                    params["quality"] = 95
                pil.save(out_path, **params)
                saved += 1
            except Exception:
                pass
            finally:
                pbar.update(1)

        return (saved, output_folder)

class ThresholdImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input edge maps to binarize"}),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Threshold from 0.0 to 1.0 — pixels below are black, above are white"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("binarized_images",)
    FUNCTION = "threshold_images"
    CATEGORY = "Image Processing"

    def threshold_images(self, images, threshold):
        images_np = convert_tensor_to_numpy(images)  # shape: [B, H, W, 3], dtype: uint8 or float32

        binarized_images = []
        threshold_255 = threshold * 255.0

        for image in images_np:
            # If image is in float [0, 1], scale it to [0, 255]
            if image.dtype == np.float32 and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            # Grayscale using luminosity method
            grayscale = (0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2])

            # Apply threshold
            binary = (grayscale >= threshold_255).astype(np.uint8) * 255

            # Convert back to RGB
            binarized_rgb = np.stack([binary]*3, axis=-1).astype(np.uint8)
            binarized_images.append(binarized_rgb)

        binarized_np = np.stack(binarized_images, axis=0)
        binarized_tensor = convert_numpy_to_tensor(binarized_np)

        return (binarized_tensor,)


class StringToFloatList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "numbers": ("STRING", {"default": "3,4.3,5", "tooltip": "Comma-separated numbers or JSON list, e.g., 1,2.5,3 or [1,2.5,3]"}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float_list",)
    FUNCTION = "parse"
    CATEGORY = "Utilities/Parsing"

    def parse(self, numbers: str):
        # Try JSON first
        parsed: list = []
        try:
            obj = json.loads(numbers)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (int, float)):
                        parsed.append(float(item))
                    elif isinstance(item, str) and item.strip() != "":
                        try:
                            parsed.append(float(item.strip()))
                        except Exception:
                            continue
                return (parsed,)
        except Exception:
            pass

        # Fallback: flexible delimiter parsing
        s = numbers.strip()
        # Normalize brackets and separators
        for ch in "[](){}":
            s = s.replace(ch, "")
        s = s.replace(";", ",")
        # If there are no commas, split on whitespace
        tokens = [t for t in s.split(",")] if "," in s else s.split()
        for token in tokens:
            t = token.strip()
            if t == "":
                continue
            try:
                parsed.append(float(t))
            except Exception:
                # Skip non-numeric tokens
                continue

        return (parsed,)

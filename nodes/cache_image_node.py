import os
import hashlib
import random
from PIL import Image, ImageSequence, ImageOps
from io import BytesIO
import numpy as np
import torch
import requests
import mimetypes
import re
import glob
from threading import Lock
import portalocker  # Cross-platform file locking

# Define the directory where images will be cached
CACHE_DIR = "cached_images"
# Path to access count file
ACCESS_COUNT_FILE = os.path.join(CACHE_DIR, "access_counts.txt")

# Ensure the cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

class CacheImageNode:
    _access_counts_lock = Lock()  # Class-level lock for thread safety

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"multiline": False, "dynamicPrompts": False})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_and_cache_image"
    CATEGORY = "image"

    def __init__(self):
        self.internal_access_counts = {}
        self.access_counts = self.load_access_counts()

    def load_access_counts(self):
        # Load access counts from the file
        access_counts = {}
        if os.path.exists(ACCESS_COUNT_FILE):
            with portalocker.Lock(ACCESS_COUNT_FILE, 'r', timeout=5) as f:
                for line in f:
                    image_hash, count = line.strip().split()
                    access_counts[image_hash] = int(count)
        return access_counts

    def load_and_cache_image(self, image_url):
    if not image_url:
        raise ValueError("No image URL provided")

    # Generate a unique filename using a hash of the URL
    image_hash = hashlib.md5(image_url.encode('utf-8')).hexdigest()

    # Determine the file extension from the URL or response headers
    extension = self.get_file_extension(image_url)

    cached_image_path = os.path.join(CACHE_DIR, f"{image_hash}{extension}")

    # Check if the image is already cached
    if os.path.exists(cached_image_path):
        # Load the image and convert to tensor
        img = Image.open(cached_image_path)
        img_out, mask_out = self.pil2tensor(img)
        return (img_out, mask_out)

    # Download the image
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image from URL: {image_url}. Error: {str(e)}")

    # Update extension based on Content-Type if necessary
    content_type = response.headers.get('Content-Type', '')
    if not extension or extension == '.unknown':
        extension = mimetypes.guess_extension(content_type.split(';')[0]) or '.png'
        # Update cached image path with new extension
        cached_image_path = os.path.join(CACHE_DIR, f"{image_hash}{extension}")

    # Save the image to the cache directory
    if 'image' in content_type:
        try:
            image = Image.open(BytesIO(response.content))
            image.save(cached_image_path)
        except Exception as e:
            # If PIL cannot open the image, save it as binary
            with open(cached_image_path, "wb") as f:
                f.write(response.content)
    else:
        # Save non-image files as they are
        with open(cached_image_path, "wb") as f:
            f.write(response.content)

    # Occasionally check used space and purge if necessary (0.1% of the requests)
    if random.random() < 0.001:
        self.update_access_counts()
        self.purge_cache()

    # Occasionally update the access count file (5% of the requests)
    if random.random() < 0.05:
        self.update_access_counts()

    # Load the image and convert to tensor
    img = Image.open(cached_image_path)
    img_out, mask_out = self.pil2tensor(img)
    
    return (img_out, mask_out)

    def get_file_extension(self, image_url):
        # Try to get the file extension from the URL
        parsed_url = re.sub(r'\?.*$', '', image_url)  # Remove query parameters
        root, ext = os.path.splitext(parsed_url)
        if ext:
            return ext
        else:
            return '.unknown'  # Will attempt to guess later

    def pil2tensor(self, img):
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None, ]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros(image.shape[1:3], dtype=torch.float32)
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    def purge_cache(self):
        # Calculate the total size of the cache directory
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(CACHE_DIR)
            for filename in filenames
            if filename != os.path.basename(ACCESS_COUNT_FILE)
        )

        # If the cache size exceeds a certain limit (e.g., 50GB), purge less accessed files
        CACHE_LIMIT = 50 * 1024 * 1024 * 1024  # 50GB
        if total_size > CACHE_LIMIT:
            # Sort files by access count (ascending)
            with self._access_counts_lock:
                sorted_files = sorted(self.access_counts.items(), key=lambda x: x[1])

            freed_space = 0
            removed_hashes = []

            for image_hash, _ in sorted_files:
                # Find the exact file with known extension
                pattern = os.path.join(CACHE_DIR, f"{image_hash}.*")
                file_list = glob.glob(pattern)
                if not file_list:
                    continue  # File may have been deleted already
                file_path = file_list[0]  # Assuming only one file per hash
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    freed_space += file_size
                    removed_hashes.append(image_hash)

                    # Stop purging if enough space has been freed
                    if total_size - freed_space <= CACHE_LIMIT:
                        break

            # Remove purged entries from access counts
            with self._access_counts_lock:
                for image_hash in removed_hashes:
                    self.access_counts.pop(image_hash, None)
                self.save_access_counts()

    def save_access_counts(self):
        # Save access counts to the file
        with portalocker.Lock(ACCESS_COUNT_FILE, 'w', timeout=5) as f:
            for image_hash, count in self.access_counts.items():
                f.write(f"{image_hash} {count}\n")

    def update_access_counts(self):
        # Merge internal access counts with global access counts
        with self._access_counts_lock:
            for image_hash, delta in self.internal_access_counts.items():
                self.access_counts[image_hash] = self.access_counts.get(image_hash, 0) + delta
            self.internal_access_counts.clear()
            self.save_access_counts()

    def __del__(self):
        # Save access counts to file when the node is deleted
        self.update_access_counts()

# Register the node
NODE_CLASS_MAPPINGS = {
    "CacheImageNode": CacheImageNode
}

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
import fcntl
import time
import contextlib
from threading import Lock

# Use RunPod's shared storage path
CACHE_DIR = "/workspace/ComfyUI/cached_images"  # Adjust this path as needed
ACCESS_COUNT_FILE = os.path.join(CACHE_DIR, "access_counts.txt")

@contextlib.contextmanager
def file_lock(filename):
    """Distributed file locking mechanism"""
    lock_file = f"{filename}.lock"
    with open(lock_file, 'w') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            try:
                os.remove(lock_file)
            except:
                pass

def ensure_directory_exists(directory):
    """Ensure directory exists with proper error handling"""
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        raise

def write_atomic(file_path, content, mode='wb'):
    """Write file atomically to prevent partial writes"""
    temp_path = f"{file_path}.temp"
    try:
        with open(temp_path, mode) as f:
            f.write(content)
        os.rename(temp_path, file_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def cleanup_stale_locks():
    """Clean up stale lock files"""
    lock_files = glob.glob(os.path.join(CACHE_DIR, "*.lock"))
    for lock_file in lock_files:
        try:
            if time.time() - os.path.getctime(lock_file) > 3600:  # 1 hour
                os.remove(lock_file)
        except Exception:
            pass

# Ensure the cache directory exists
ensure_directory_exists(CACHE_DIR)

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
        if random.random() < 0.01:  # 1% chance
            cleanup_stale_locks()

    def load_access_counts(self):
        access_counts = {}
        if os.path.exists(ACCESS_COUNT_FILE):
            with file_lock(ACCESS_COUNT_FILE):
                with open(ACCESS_COUNT_FILE, 'r') as f:
                    for line in f:
                        try:
                            image_hash, count = line.strip().split()
                            access_counts[image_hash] = int(count)
                        except ValueError:
                            continue
        return access_counts

    def save_access_counts(self):
        with file_lock(ACCESS_COUNT_FILE):
            write_atomic(ACCESS_COUNT_FILE, 
                        '\n'.join(f"{image_hash} {count}" 
                                 for image_hash, count in self.access_counts.items()),
                        mode='w')

    def load_and_cache_image(self, image_url):
        if not image_url:
            raise ValueError("No image URL provided")

        image_hash = hashlib.md5(image_url.encode('utf-8')).hexdigest()
        cached_files = glob.glob(os.path.join(CACHE_DIR, f"{image_hash}.*"))
        
        if cached_files:
            cached_image_path = cached_files[0]
            with file_lock(cached_image_path):
                img = Image.open(cached_image_path)
                img_out, mask_out = self.pil2tensor(img)
            
            with self._access_counts_lock:
                self.internal_access_counts[image_hash] = self.internal_access_counts.get(image_hash, 0) + 1
            
            return (img_out, mask_out)

        extension = self.get_file_extension(image_url)
        cached_image_path = os.path.join(CACHE_DIR, f"{image_hash}{extension}")

        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download image from URL: {image_url}. Error: {str(e)}")

        content_type = response.headers.get('Content-Type', '')
        if not extension or extension == '.unknown':
            extension = mimetypes.guess_extension(content_type.split(';')[0]) or '.png'
            cached_image_path = os.path.join(CACHE_DIR, f"{image_hash}{extension}")

        with file_lock(cached_image_path):
            if 'image' in content_type:
                try:
                    image = Image.open(BytesIO(response.content))
                    image.save(cached_image_path)
                except Exception:
                    write_atomic(cached_image_path, response.content)
            else:
                write_atomic(cached_image_path, response.content)

        if random.random() < 0.001:  # 0.1% chance
            self.update_access_counts()
            self.purge_cache()

        if random.random() < 0.05:  # 5% chance
            self.update_access_counts()

        with file_lock(cached_image_path):
            img = Image.open(cached_image_path)
            img_out, mask_out = self.pil2tensor(img)
        
        with self._access_counts_lock:
            self.internal_access_counts[image_hash] = self.internal_access_counts.get(image_hash, 0) + 1
        
        return (img_out, mask_out)

    def get_file_extension(self, image_url):
        parsed_url = re.sub(r'\?.*$', '', image_url)
        root, ext = os.path.splitext(parsed_url)
        return ext.lower() if ext else '.unknown'

    def pil2tensor(self, img):
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
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
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(CACHE_DIR)
            for filename in filenames
            if not filename.endswith('.lock') and filename != os.path.basename(ACCESS_COUNT_FILE)
        )

        CACHE_LIMIT = 50 * 1024 * 1024 * 1024  # 50GB
        if total_size > CACHE_LIMIT:
            with self._access_counts_lock:
                sorted_files = sorted(self.access_counts.items(), key=lambda x: x[1])

            freed_space = 0
            removed_hashes = []

            for image_hash, _ in sorted_files:
                pattern = os.path.join(CACHE_DIR, f"{image_hash}.*")
                file_list = glob.glob(pattern)
                if not file_list:
                    continue
                
                file_path = file_list[0]
                with file_lock(file_path):
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        freed_space += file_size
                        removed_hashes.append(image_hash)

                if total_size - freed_space <= CACHE_LIMIT:
                    break

            with self._access_counts_lock:
                for image_hash in removed_hashes:
                    self.access_counts.pop(image_hash, None)
                self.save_access_counts()

    def update_access_counts(self):
        with self._access_counts_lock:
            for image_hash, delta in self.internal_access_counts.items():
                self.access_counts[image_hash] = self.access_counts.get(image_hash, 0) + delta
            self.internal_access_counts.clear()
            self.save_access_counts()

    def __del__(self):
        try:
            self.update_access_counts()
        except:
            pass

# Register the node
NODE_CLASS_MAPPINGS = {
    "CacheImageNode": CacheImageNode
}

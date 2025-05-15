import httpx

import io
from PIL import Image
from pathlib import Path

class WebImage:

    def bytes2image(image_bytes: bytes):
        return Image.open(io.BytesIO(image_bytes))
    
    def jpeg2png(image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True, compress_level=9)
        return buffer.getvalue()

    def compress_png(
        image_bytes: bytes,
        max_size_mb: float = 16.0,
        target_mode: str = "RGBA",
        min_scale_ratio: float = 0.1
    ) -> bytes:
        """
        将图像(通常是PNG)在保持 PNG 格式的前提下进行压缩。
        - 强制输出通道模式为 target_mode (可以是 RGBA、LA、L 等)。
        - 若无损压缩仍超出 max_size_mb，则通过二分搜索缩小分辨率，直到文件小于限制或达最小缩放比。
        返回处理后的 PNG bytes。
        
        参数说明:
        - image_bytes:     原图的二进制数据 (bytes)
        - max_size_mb:     限制大小 (单位 MB)
        - target_mode:     期望最终图像的 mode (如 "RGBA"、"LA"、"L")
        - min_scale_ratio: 最小缩放比，避免无限缩小 (默认 0.1，即 10%)
        """
        max_size = int(max_size_mb * 1024 * 1024)
        if len(image_bytes) <= max_size:
            return image_bytes
        
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != target_mode:
            image = image.convert(target_mode)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True, compress_level=9)
        data_no_resize = buffer.getvalue()
        
        if len(data_no_resize) <= max_size:
            return data_no_resize
        
        orig_width, orig_height = image.size
        
        left, right = min_scale_ratio, 1.0
        best_data = None
        best_scale = left
        
        while (right - left) > 0.01: 
            mid = (left + right) / 2.0
            
            new_width = int(orig_width * mid)
            new_height = int(orig_height * mid)
            
            resized_img = image.resize((new_width, new_height), Image.LANCZOS)
            
            temp_buffer = io.BytesIO()
            resized_img.save(temp_buffer, format="PNG", optimize=True, compress_level=9)
            temp_data = temp_buffer.getvalue()
            
            if len(temp_data) <= max_size:
                best_data = temp_data
                best_scale = mid
                left = mid
            else:
                right = mid
        
        if best_data is None:
            final_scale = min_scale_ratio
        else:
            final_scale = best_scale

        final_width = int(orig_width * final_scale)
        final_height = int(orig_height * final_scale)
        resized_img = image.resize((final_width, final_height), Image.LANCZOS)
        temp_buffer = io.BytesIO()
        resized_img.save(temp_buffer, format="PNG", optimize=True, compress_level=9)
        final_data = temp_buffer.getvalue()
        
        return final_data

    def compress_image(image_bytes: bytes, max_size_mb: float = 2.0) -> bytes:
        """
        当图像大于 max_size_mb MB 时，将其压缩到尽可能接近但小于 max_size_mb MB。
        返回处理后的 bytes。
        """

        max_size = int(max_size_mb * 1024 * 1024)
        if len(image_bytes) <= max_size:
            return image_bytes

        image = Image.open(io.BytesIO(image_bytes))

        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        left, right = 1, 95
        best_quality = left
        best_data = None

        while left <= right:
            mid = (left + right) // 2

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=mid)
            data = buffer.getvalue()

            if len(data) <= max_size:
                best_quality = mid
                best_data = data
                left = mid + 1
            else:
                right = mid - 1

        if not best_data:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=1)
            best_data = buffer.getvalue()

        return best_data
        
    def __init__(self, country_f, country_t):
        self.country_from = country_f
        self.country_to = country_t

        self.cache_path = f"./download_images/"
        if not Path.exists(Path(self.cache_path)):
            Path(self.cache_path).mkdir(parents=True, exist_ok=True)

    def url2path(self, url):
        # https://storage.googleapis.com/image-transcreation/part1/india/beverages_kingfisher-beer.jpg
        return f"{self.cache_path}/{url.split('storage.googleapis.com/')[-1]}"
    
    def read_image_fp(self, url):
        # Note: no compression
        local_path = self.url2path(url)
        if not Path.exists(Path(local_path)):
            # Create a local directory based on local_path
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                image = httpx.get(url).content
            except Exception as e:
                return None
            with open(local_path, "wb") as f:
                f.write(image)
        return local_path

    def read_image(self, url, max_size_mb: float = 2.0):
        local_path = self.read_image_fp(url)
        with open(local_path, "rb") as f:
            image = f.read()

        if max_size_mb > 0:
            image = WebImage.compress_image(image, max_size_mb)
        return WebImage.bytes2image(image)
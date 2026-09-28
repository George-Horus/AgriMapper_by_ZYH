"""
This script is used for automatic extraction of wheat planting regions from multispectral remote sensing images. The main steps include:

1. Read the original NDVI image:
   - Supports NDVI index images in GeoTIFF format with georeferencing.

2. Automatic threshold segmentation (Otsu algorithm):
   - Uses the Otsu method to adaptively calculate the optimal threshold from the NDVI value histogram, requiring no manual setting.

3. Mask generation:
   - Generates a binary mask where NDVI > threshold, where 1 represents wheat regions, and 0 represents background regions.

4. Mask alignment and application:
   - Resamples and aligns the mask to the spatial reference system of the original multi-band image, then applies the mask to retain only the wheat regions’ multi-band information, setting other areas to 0.

5. Result outputs:
   - Saves the binary mask image (GeoTIFF)
   - Saves the masked multi-band image (GeoTIFF)
"""

import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from skimage.filters import threshold_otsu  # For automatic threshold calculation

def extract_canopy(multiband_path, ndvi_tif_path, output_mask_path, output_masked_tif):
    """Create an Otsu NDVI mask and apply it to a multispectral image."""
    with rasterio.open(multiband_path) as src:
        bands = src.read()
        meta = src.meta.copy()
        target_height, target_width = src.height, src.width
        target_transform = src.transform
        target_crs = src.crs

    with rasterio.open(ndvi_tif_path) as ndvi_src:
        ndvi_data = ndvi_src.read(1).astype("float32")
        ndvi_resampled = np.empty((target_height, target_width), dtype="float32")
        reproject(
            source=ndvi_data,
            destination=ndvi_resampled,
            src_transform=ndvi_src.transform,
            src_crs=ndvi_src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )

    ndvi_valid = ndvi_resampled[(ndvi_resampled >= 0) & (ndvi_resampled <= 1)]
    if ndvi_valid.size == 0:
        raise ValueError("No NDVI pixels in the expected [0, 1] range.")
    otsu_threshold = threshold_otsu(ndvi_valid)
    print(f"Otsu NDVI threshold: {otsu_threshold:.4f}")

    ndvi_mask = ndvi_resampled > otsu_threshold
    mask_meta = meta.copy()
    mask_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
    with rasterio.open(output_mask_path, "w", **mask_meta) as dst:
        dst.write(ndvi_mask.astype("uint8"), 1)

    masked_bands = np.where(ndvi_mask[np.newaxis, :, :], bands, 0)
    meta.update({"dtype": "float32", "nodata": 0})
    with rasterio.open(output_masked_tif, "w", **meta) as dst:
        dst.write(masked_bands.astype("float32"))

    print(f"NDVI mask saved to: {output_mask_path}")
    print(f"Masked image saved to: {output_masked_tif}")
    return output_mask_path, output_masked_tif


if __name__ == "__main__":
    extract_canopy(
        r"D:\Zyh\20250421comp.tif",
        r"D:\Zyh\20250421_index_ndvi.tif",
        r"D:\Zyh\20250421_ndvi_mask.tif",
        r"D:\Zyh\20250421_wheat_masked_output.tif",
    )

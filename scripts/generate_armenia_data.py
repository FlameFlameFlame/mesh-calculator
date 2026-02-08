"""
Generate input data for Armenia mesh network region.

Cities:
- Priority 1: Yerevan, Gyumri, Vanadzor
- Priority 2: Hrazdan, Dilijan, Sevan

Downloads:
- Roads from OpenStreetMap (Overpass API)
- Elevation from SRTM (NASA 30m)
"""

import json
import struct
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
import requests

DATA_DIR = Path(__file__).parent.parent / "data"

# Bounding box for the region (covers Gyumri to Sevan/Dilijan)
BBOX = {
    "south": 40.05,
    "north": 40.95,
    "west": 43.65,
    "east": 45.10,
}


def download_roads():
    """Download major roads from OpenStreetMap via Overpass API."""
    print("Downloading roads from OpenStreetMap...")

    # Overpass query for major roads in the region
    query = f"""
    [out:json][timeout:120];
    (
      way["highway"~"^(motorway|trunk|primary|secondary)$"]
        ({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});
    );
    out body;
    >;
    out skel qt;
    """

    url = "https://overpass-api.de/api/interpreter"
    resp = requests.post(url, data={"data": query}, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    # Build node lookup
    nodes = {}
    for elem in data["elements"]:
        if elem["type"] == "node":
            nodes[elem["id"]] = (elem["lon"], elem["lat"])

    # Build GeoJSON features from ways
    features = []
    for elem in data["elements"]:
        if elem["type"] == "way":
            coords = []
            for nid in elem.get("nodes", []):
                if nid in nodes:
                    coords.append(list(nodes[nid]))
            if len(coords) >= 2:
                tags = elem.get("tags", {})
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                    "properties": {
                        "name": tags.get("name", tags.get("name:en", "")),
                        "type": tags.get("highway", "road"),
                        "osm_id": elem["id"],
                    },
                })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    output_path = DATA_DIR / "armenia_roads.geojson"
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"  Saved {len(features)} road segments to {output_path}")
    return output_path


def download_srtm_tile(lat, lon):
    """Download a single SRTM tile (.hgt) from NASA/USGS mirror."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    tile_name = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
    filename = f"{tile_name}.hgt"

    # Try multiple SRTM mirrors
    urls = [
        f"https://elevation-tiles-prod.s3.amazonaws.com/skadi/{tile_name[:3]}/{filename}.gz",
        f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/",
    ]

    print(f"  Downloading SRTM tile {tile_name}...")

    # Try the AWS elevation tiles (gzipped .hgt)
    url = urls[0]
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        import gzip
        hgt_data = gzip.decompress(resp.content)
        print(f"    Downloaded {len(hgt_data)} bytes from AWS")
        return tile_name, hgt_data
    except Exception as e:
        print(f"    AWS mirror failed: {e}")

    # Fallback: try viewfinderpanoramas.org
    url = f"http://viewfinderpanoramas.org/dem3/{tile_name[:3]}.zip"
    try:
        print(f"    Trying viewfinderpanoramas.org...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                if name.upper().endswith(f"{tile_name}.HGT") or name.upper() == f"{tile_name.upper()}.HGT":
                    hgt_data = zf.read(name)
                    print(f"    Downloaded {len(hgt_data)} bytes from viewfinderpanoramas")
                    return tile_name, hgt_data
            # Try any .hgt file that matches
            for name in zf.namelist():
                if tile_name.upper() in name.upper() and name.upper().endswith(".HGT"):
                    hgt_data = zf.read(name)
                    print(f"    Downloaded {len(hgt_data)} bytes ({name})")
                    return tile_name, hgt_data
            print(f"    Available files in zip: {zf.namelist()}")
    except Exception as e:
        print(f"    viewfinderpanoramas failed: {e}")

    raise RuntimeError(f"Could not download SRTM tile {tile_name}")


def hgt_to_array(hgt_data):
    """Convert raw .hgt bytes to a numpy array."""
    # SRTM3 (3 arc-second): 1201x1201
    # SRTM1 (1 arc-second): 3601x3601
    size = len(hgt_data)
    if size == 1201 * 1201 * 2:
        dim = 1201
    elif size == 3601 * 3601 * 2:
        dim = 3601
    else:
        raise ValueError(f"Unexpected HGT file size: {size} bytes")

    arr = np.frombuffer(hgt_data, dtype=">i2").reshape((dim, dim)).astype(np.float32)
    # Replace voids (-32768) with 0
    arr[arr < -1000] = 0
    return arr


def create_elevation_geotiff():
    """Download SRTM tiles and merge into a single GeoTIFF."""
    print("Downloading elevation data...")

    # We need tiles covering lat 40-41, lon 43-46
    # SRTM tiles are named by SW corner: N40E043, N40E044, N40E045
    tiles_needed = [
        (40, 43),
        (40, 44),
        (40, 45),
    ]

    tile_arrays = []
    tile_transforms = []

    for lat, lon in tiles_needed:
        tile_name, hgt_data = download_srtm_tile(lat, lon)
        arr = hgt_to_array(hgt_data)
        dim = arr.shape[0]

        # SRTM tile covers [lon, lon+1] x [lat, lat+1]
        # Top-left is (lon, lat+1), pixel size = 1/(dim-1) degree
        pixel_size = 1.0 / (dim - 1)
        transform = from_bounds(lon, lat, lon + 1, lat + 1, dim, dim)

        tile_arrays.append(arr)
        tile_transforms.append((transform, dim, lat, lon))

    # Write individual tiles as temporary rasters, then merge
    import tempfile
    temp_files = []

    for i, (arr, (transform, dim, lat, lon)) in enumerate(zip(tile_arrays, tile_transforms)):
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        temp_files.append(tmp.name)

        with rasterio.open(
            tmp.name,
            "w",
            driver="GTiff",
            height=dim,
            width=dim,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(arr, 1)

    # Merge tiles
    datasets = [rasterio.open(f) for f in temp_files]
    merged_arr, merged_transform = merge(datasets)

    for ds in datasets:
        ds.close()

    # Crop to our bounding box
    output_path = DATA_DIR / "armenia_elevation.tif"

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=merged_arr.shape[1],
        width=merged_arr.shape[2],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=merged_transform,
        compress="deflate",
    ) as dst:
        dst.write(merged_arr[0], 1)

    # Clean up temp files
    import os
    for f in temp_files:
        os.unlink(f)

    print(f"  Saved elevation GeoTIFF to {output_path}")
    print(f"  Shape: {merged_arr.shape[1:]} pixels")
    return output_path


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download roads
    roads_path = download_roads()

    # Download elevation
    elev_path = create_elevation_geotiff()

    print("\nAll data files generated:")
    print(f"  Sites:     data/armenia_sites.geojson")
    print(f"  Boundary:  data/armenia_boundary.geojson")
    print(f"  Roads:     {roads_path.relative_to(DATA_DIR.parent)}")
    print(f"  Elevation: {elev_path.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()

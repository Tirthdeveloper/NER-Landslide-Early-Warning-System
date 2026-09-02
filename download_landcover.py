import os
import requests
import geopandas as gpd

from tqdm import tqdm
from shapely.geometry import box


# ==========================================
# CONFIGURATION
# ==========================================

# North East India bounding box
WEST = 88
SOUTH = 21
EAST = 98
NORTH = 30

# Output folder
OUTPUT_FOLDER = "data/raw/landcover"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# ESA WORLDCOVER GRID
# ==========================================

grid_url = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
    "esa_worldcover_grid.geojson"
)


print("Downloading WorldCover tile grid...")

grid_file = "worldcover_grid.geojson"

response = requests.get(grid_url)

if response.status_code == 200:

    with open(grid_file, "wb") as file:
        file.write(response.content)

    print("✅ Grid downloaded")

else:

    print("❌ Grid download failed")
    print(response.status_code)
    exit()


# ==========================================
# READ TILE GRID
# ==========================================

grid = gpd.read_file(grid_file)


# Create NER bounding box
ner_bbox = box(
    WEST,
    SOUTH,
    EAST,
    NORTH
)


# Select tiles intersecting NER
tiles = grid[
    grid.geometry.intersects(ner_bbox)
]


print(f"\nTiles required: {len(tiles)}")


# ==========================================
# DOWNLOAD WORLDCOVER TILES
# ==========================================

for _, row in tqdm(
    tiles.iterrows(),
    total=len(tiles),
    desc="Downloading Land Cover"
):

    tile_name = row["ll_tile"]

    filename = (
        f"ESA_WorldCover_10m_2021_v200_"
        f"{tile_name}_Map.tif"
    )

    url = (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
        f"v200/2021/map/{filename}"
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    # Skip already downloaded files
    if os.path.exists(output_path):
        print(f"Already exists: {filename}")
        continue

    response = requests.get(
        url,
        stream=True
    )

    if response.status_code == 200:

        total_size = int(
            response.headers.get(
                "content-length",
                0
            )
        )

        with open(output_path, "wb") as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    file.write(chunk)

        print(f"✅ Downloaded: {filename}")

    else:

        print(
            f"❌ Failed: {filename}"
        )


print("\n✅ WorldCover download completed!")
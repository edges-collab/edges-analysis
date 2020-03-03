# This is the GLOBAL configuration for all of edges-analysis.
import os

# edges_folder: something like /data5/edges/data/
# MRO_folder: path on enterprise containing files from field containing weather/metadata/etc.
# home_folder: should be same as edges_folder: point to highest_level edges drive. houses data and outputs.
config = {"edges_folder": "", "MRO_folder": "", "home_folder": os.path.expanduser("~")}

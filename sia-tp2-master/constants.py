# Main directories for execution
CONFIGS_DIR = "configs"            # Folder where algorithm configurations are stored
INPUT_IMAGES_DIR = "input_images"  # Folder containing the original images to compress
OUTPUT_IMAGES_DIR = "ga_output"    # Folder where results are saved (generated images, logs, etc.)

# Image scaling factor
SCALE_FACTOR = 3  # Scale applied to resize input images (reduce teh size)

# Transparency parameters (alpha channel of the triangles --> min alpha avoids fully transparent triangles)
MIN_ALPHA = 50    # Minimum transparency (0 = fully transparent, 255 = fully opaque)
MAX_ALPHA = 255   # Maximum transparency

# Constraints on triangle area
MIN_AREA = 1000   # Minimum allowed area for a triangle
MAX_AREA = 10000  # Maximum allowed area for a triangle

# Randomness seed
SEED = 43  # Fixes the random generator to make runs reproducible

# Color representation configuration
RGB = True   # If True → use RGB colors (red, green, blue)
             # If False, HSV (hue, saturation, value)

# Delta mutation flag
DELTA = False  # If True → mutations apply incremental changes (deltas)
               # If False → random values are generated

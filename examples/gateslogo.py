import diffractsim
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

diffractsim.set_backend("CPU")  # Change to "CUDA" for GPU acceleration

from diffractsim import PolychromaticField, ApertureFromImage, cf, mm, cm

# Create the polychromatic field
F = PolychromaticField(
    spectrum=1.5 * cf.illuminant_d65,
    extent_x=20 * mm,
    extent_y=20 * mm,
    Nx=1600,
    Ny=1600,
)

# Add the aperture
F.add(ApertureFromImage("./apertures/gates_logo.jpg", image_size=(15 * mm, 15 * mm), simulation=F))

# Create the 'test' folder if it doesn't exist
os.makedirs('test', exist_ok=True)

for i in range(150):
    # Propagate the field
    F.propagate(z=1*cm)
    rgb = F.get_colors()
    # Plot the colors
    F.plot_colors(rgb, xlim=[-10*mm, 10*mm], ylim=[-10*mm, 10*mm], units=mm, dark_background=True, savefile=f'test/neo_{i+1:03d}')

# Define the path to the directory containing the PNG files
image_folder = 'test'
# Get all PNG files in the folder and sort them numerically
images = sorted(glob.glob(os.path.join(image_folder, '*.png')), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

# Create a list to hold the frames
frames = []

# Open each image and append it to the frames list
for image in images:
    new_frame = Image.open(image)
    frames.append(new_frame)

# Save into a GIF file that loops forever with a duration of 1000 ms (1 second) per frame
gif_path = 'test/output.gif'
frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=1000, loop=0)

print(f"GIF saved as {gif_path}")

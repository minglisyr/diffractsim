import diffractsim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

diffractsim.set_backend("CPU")

from diffractsim import PolychromaticField, ApertureFromImage, cf, mm, cm

# Set up dark theme for matplotlib
plt.style.use('dark_background')

F = PolychromaticField(
    spectrum=1.5 * cf.illuminant_d65,
    extent_x=20 * mm,
    extent_y=20 * mm,
    Nx=1600,
    Ny=1600,
)

F.add(ApertureFromImage("./apertures/gates_logo.jpg", image_size=(15 * mm, 15 * mm), simulation=F))

# Create a figure and axis for the animation with dark background
fig, ax = plt.subplots(facecolor='black')
ax.set_facecolor('black')

# Initialize the plot
im = ax.imshow(np.zeros((1600, 1600, 3)), extent=[-10*mm, 10*mm, -10*mm, 10*mm])

# Customize axis and title colors
ax.tick_params(colors='white')
ax.set_xlabel('X (mm)', color='white')
ax.set_ylabel('Y (mm)', color='white')

# Animation update function
def update(frame):
    z = frame * 50 * cm
    F.propagate(z=z)
    rgb = F.get_colors()
    im.set_array(rgb)
    
    # White text for title
    ax.set_title(f"Z = {z/cm:.0f} cm", color='white')

    return [im]

# Create the animation
anim = FuncAnimation(fig, update, frames=4, interval=1000, blit=True)

# Save the animation as a gif
anim.save("../images/gates_logo_animated.gif", writer="pillow")

plt.close(fig)

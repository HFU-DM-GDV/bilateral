# Bilateral filtering illustrated

Demonstration of Bilateral Filter Kernel Computation and Visualization. 

It provides an interactive plot where clicking on an intensity patch updates the displayed bilateral filter kernel based on the selected pixel's intensity. It is inspired by the illustration from [C. Tomasi and R. Manduchi, "Bilateral filtering for gray and color images," Sixth International Conference on Computer Vision (IEEE Cat. No.98CH36271), Bombay, India, 1998, pp. 839-846, doi: 10.1109/ICCV.1998.710815.,](https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf)

Vibe coded by: Uwe Hahne

Date: November 2025

## Installation

Make sure you have the required libraries installed. You can install them using pip:

```bash
pip install opencv-python
pip install matplotlib
```

The code has been developed and tested with Python 3.12.2 using OpenCV 4.12.0.88 and Matplotlib 3.10.7.

## Usage

Run the script using Python:

```bash
python bilateralFilter.py
```

Click on the intensity patch to see how the bilateral filter kernel changes based on the selected pixel's intensity.


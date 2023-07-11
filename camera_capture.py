import datetime
import numpy as np
from multiprocessing import Pool, freeze_support
import dv_processing as dv
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Show a preview of an iniVation event camera input.')

args = parser.parse_args()

data = []

# Open the camera
reader = dv.io.MonoCameraRecording(r"C:\Users\LATITUDE\Downloads\user01_fluorescent.aedat4")

positive_color = [183,93,0]

negative_color = [43,43,43]

# Initialize visualizer instance which generates event data preview
visualizer = dv.visualization.EventVisualizer(reader.getEventResolution())

# Create the preview window
cv.namedWindow("Binary", cv.WINDOW_NORMAL)

def transform_tile(tile):
    tile = tile[:128, :]

    tile = tile[:, :128]

    positive_color_matrix = np.all(tile == positive_color, axis = 2)

    negative_color_matrix = 0 - np.all(tile == negative_color, axis = 2)
    
    transformed_tile = positive_color_matrix + negative_color_matrix

    return transformed_tile

def preview_events(event_slice):
    global data
    # Show the accumulated image
    original = visualizer.generateImage(event_slice)
    image = np.uint8((transform_tile(original) + 1) * 127)
    data.append(image)
    data_np = np.array(data)
    print(np.shape(data_np))
    cv.imshow("Binary", image)
    cv.imshow("data_np", data_np[-1])
    
    # cv.imshow("Preview", visualizer.generateImage(event_slice))
    # print(visualizer.generateImage(event_slice))
    cv.waitKey(2)


# Create an event slicer, this will only be used events only camera
slicer = dv.EventStreamSlicer()
slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=1000), preview_events)

# start read loop
while reader.isRunning():
    # Get events
    events = reader.getNextEventBatch()

    # If no events arrived yet, continue reading
    if events is not None:
        slicer.accept(events)

data_np = np.array(data)
print(data_np.shape)
# np.savez('maher_282_np.npz', data_np)
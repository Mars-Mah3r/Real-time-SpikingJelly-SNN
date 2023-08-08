import datetime
import numpy as np
from multiprocessing import Pool, freeze_support
import dv_processing as dv
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Show a preview of an iniVation event reader input.')

args = parser.parse_args()

# Open the reader
reader = dv.io.MonoCameraRecording(r"C:\Users\LATITUDE\Downloads\user01_fluorescent.aedat4") # the .aedat4 file

maher_t = []    # list for timestamps
maher_x = []    # list for x-coordinates
maher_y = []    # list for y-coordinates
maher_p = []    # list for polarities

# in this particular case, dv's visualiser maps positive events and negative events to their respective colours using the following values for the RGB channels
positive_color = [183,93,0]

negative_color = [43,43,43]

# Initialize visualizer instance which generates event data preview
visualizer = dv.visualization.EventVisualizer(reader.getEventResolution())

# Create the preview window
cv.namedWindow("Binary", cv.WINDOW_NORMAL)

def save(t, x, y, p):
    np.savez(r'C:\Users\LATITUDE\Documents\KURF\maher.npz', t = np.array(t), x = np.array(x), y = np.array(y), p = np.array(p)) # where one wishes to save the .npz file containing the raw event data

def transform_tile(tile):
    # This function transorms a 3-channel matrix with values ranging from 0 to 255 to a single channel trinary matrix (values of -1, 0 or 1)

    # Keep only the top left 128 * 128 pixels
    tile = tile[:128, :]

    tile = tile[:, :128]

    # Transform RGB to grayscale by mapping positive colours within the generated image/matrix to 1 and negative colours to -1

    # filter for positive
    positive_color_matrix = np.all(tile == positive_color, axis = 2)

    # filter for negative
    negative_color_matrix = 0 - np.all(tile == negative_color, axis = 2)
    
    # add these 2 binary matrices together
    transformed_tile = positive_color_matrix + negative_color_matrix

    transformed_tile = np.array(transformed_tile, dtype='uint8')
    new_width = 64
    new_height = 64

    transformed_tile = cv.resize(transformed_tile, (new_width, new_height), interpolation=cv.INTER_LINEAR)

    return transformed_tile

def preview_events(event_slice):
    # Show the accumulated image
    original = visualizer.generateImage(event_slice)

    ''' Transform RGB image and then add 1 to the resulting matrix- thus the entries of the matrix have values of 0, 1 or 2. 
        Having done so, mltiply through by 127, yielding a matrix with entries that only ever take on the values of 0, 127 or 254.
        The image is now grayscale.'''
    image = np.uint8((transform_tile(original) + 1) * 127)

    global maher_t
    global maher_x
    global maher_y
    global maher_p
    # append to the appropriate information to the appropriate attribute
    # for event in event_slice:
    #     maher_t.append(event.timestamp())
    #     maher_x.append(event.x())
    #     maher_y.append(event.y())
    #     maher_p.append(event.polarity())

    # Show the grayscale image
    cv.imshow("Binary", image)
    cv.waitKey(2)

# Create an event slicer, this will only be used events only reader
slicer = dv.EventStreamSlicer()
slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=1000), preview_events) # increase delta to playback faster (more events in a given slice)

# start read loop
while reader.isRunning():
    # Get events
    events = reader.getNextEventBatch()

    # If no events arrived yet, continue reading
    if events is not None:
        slicer.accept(events)

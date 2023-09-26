
import datetime
import numpy as np
from multiprocessing import Pool, freeze_support
import dv_processing as dv
import cv2 as cv
import argparse
import torch.nn as nn
import torch

# in this particular case, dv's visualiser maps positive events and negative events to their respective colours using the following values for the RGB channels
positive_color = [183,93,0]

negative_color = [43,43,43]

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

    transformed_tile = np.array(transformed_tile)
    
    m = nn.MaxPool2d(2)

    transformed_tile = torch.unsqueeze(torch.unsqueeze(torch.tensor(transformed_tile),0),0)

    print(transformed_tile.shape)

    transformed_tile = m(transformed_tile.float())

    transformed_tile = torch.squeeze(torch.squeeze(transformed_tile,0),0)

    return transformed_tile

def readstream():
    parser = argparse.ArgumentParser(description='Show a preview of an iniVation event capture input.')

    args = parser.parse_args()

    capture = dv.io.CameraCapture()
    visualizer = dv.visualization.EventVisualizer(capture.getEventResolution())
    
    # Create the preview window
    cv.namedWindow("Binary", cv.WINDOW_NORMAL)

    def preview_events(event_slice):
        # Show the accumulated image
        original = visualizer.generateImage(event_slice)

        ''' Transform RGB image and then add 1 to the resulting matrix- thus the entries of the matrix have values of 0, 1 or 2. 
            Having done so, multiply through by 127, yielding a matrix with entries that only ever take on the values of 0, 127 or 254.
            The image is now grayscale.'''
        image = np.uint8((transform_tile(original) + 1) * 127)

        # Show the grayscale image
        cv.imshow("Binary", image)
        cv.waitKey(2)
        return image

    # Create an event slicer, this will only be used events only capture
    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=100), preview_events) # increase delta to playback faster (more events in a given slice)

    # start read loop
    while capture.isRunning():
        # Get events
        events = capture.getNextEventBatch()

        # If no events arrived yet, continue reading
        if events is not None:
            slicer.accept(events)

def readIBM():
    parser = argparse.ArgumentParser(description='Show a preview of an iniVation event capture input.')

    args = parser.parse_args()

    reader = dv.io.MonoCameraRecording(r"C:\Users\LATITUDE\Documents\KURF\DvsGestureConverted\user01_fluorescent.aedat4")
    visualizer = dv.visualization.EventVisualizer(reader.getEventResolution())
    
    # Create the preview window
    cv.namedWindow("IBM", cv.WINDOW_NORMAL)

    def ibm(event_slice):
        # Show the accumulated image
        original = visualizer.generateImage(event_slice)

        ''' Transform RGB image and then add 1 to the resulting matrix- thus the entries of the matrix have values of 0, 1 or 2. 
            Having done so, multiply through by 127, yielding a matrix with entries that only ever take on the values of 0, 127 or 254.
            The image is now grayscale.'''
        image = np.uint8((transform_tile(original) + 1) * 127)

        # Show the grayscale image
        cv.imshow("IBM", image)
        cv.waitKey(2)
        return image

    ibmslicer = dv.EventStreamSlicer()
    ibmslicer.doEveryTimeInterval(datetime.timedelta(milliseconds=100), ibm) # increase delta to playback faster (more events in a given slice)

    while reader.isRunning():
        events = reader.getNextEventBatch()

        if events is not None:
            ibmslicer.accept(events)

if __name__ == '__main__':
    readIBM()
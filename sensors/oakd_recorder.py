import numpy as np
import cv2
import depthai


# create oak-d pipeline
pipeline = depthai.Pipeline()

# color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)

# get data from device to host
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.preview.link(xout_rgb.input)


# initialize camera
with depthai.Device(pipeline) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink

    # To consume the device results, we get output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")

    # Frame will be an image from "rgb" stream
    frame = None

    # Main host-side application loop
    while True:
        # we try to fetch the data from rgb queues. tryGet will return either the data packet or None if there isn't any
        in_rgb = q_rgb.tryGet()

        if in_rgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            frame = in_rgb.getCvFrame()

        if frame is not None:
            cv2.imshow("preview", frame)

        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break

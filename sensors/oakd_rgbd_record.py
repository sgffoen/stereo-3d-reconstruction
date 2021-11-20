import cv2
import numpy as np
import depthai as dai

# write frames
writeFrames = True
# Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
# Otherwise (False), the aligned depth is automatically upscaled to 1080p
downscaleColor = True
fps = 5
# The disparity is computed at this resolution, then upscaled to RGB resolution
# Select the camera sensor resolution: 1280×720, 1280×800, 640×400 (THE_720_P, THE_800_P, THE_400_P)
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

# Create pipeline
pipeline = dai.Pipeline()
queueNames = []

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

rgbOut = pipeline.create(dai.node.XLinkOut)
depthOut = pipeline.create(dai.node.XLinkOut)

rgbOut.setStreamName("rgb")
queueNames.append("rgb")
depthOut.setStreamName("depth")
queueNames.append("depth")

#Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setFps(fps)
if downscaleColor: camRgb.setIspScale(2, 3)
# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
camRgb.initialControl.setManualFocus(130)

left.setResolution(monoResolution)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
left.setFps(fps)
right.setResolution(monoResolution)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right.setFps(fps)

stereo.initialConfig.setConfidenceThreshold(245)
# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

# Linking
camRgb.isp.link(rgbOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.disparity.link(depthOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    device.getOutputQueue(name="rgb",   maxSize=4, blocking=False)
    device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    frameRgb = None
    frameDepth = None
    count=0
    while True:
        latestPacket = {}
        latestPacket["rgb"] = None
        latestPacket["depth"] = None

        queueEvents = device.getQueueEvents(("rgb", "depth"))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        if latestPacket["rgb"] is not None:
            frameRgb = latestPacket["rgb"].getCvFrame()
            cv2.imshow("rgb", frameRgb)

        if latestPacket["depth"] is not None:
            frameDepth = latestPacket["depth"].getFrame()
            maxDisparity = stereo.initialConfig.getMaxDisparity()
            # Optional, extend range 0..95 -> 0..255, for a better visualisation
            if 1: frameDepth = (frameDepth * 255. / maxDisparity).astype(np.uint8)
            # Optional, apply false colorization
            #if 1: frameDepth = cv2.applyColorMap(frameDepth, cv2.COLORMAP_HOT)
            frameDepth = np.ascontiguousarray(frameDepth)
            cv2.imshow("depth", frameDepth)


        # Blend when both received
        if frameRgb is not None and frameDepth is not None:
            # Need to have both frames in BGR format before blending
            if len(frameDepth.shape) < 3:
                frameDepth = cv2.cvtColor(frameDepth, cv2.COLOR_GRAY2BGR)
            # TODO add a slider to adjust blending ratio
            blended = cv2.addWeighted(frameRgb, 0.6, frameDepth, 0.4 ,0)
            cv2.imshow("rgb-depth", blended)
            skip = 3*fps # skip first 3 seconds before writing frames:
            if writeFrames is True and count>skip:
                cv2.imwrite("data/depth/framedepth%d.png" % (count-skip), frameDepth)
                cv2.imwrite("data/image/framergb%d.png" % (count-skip), frameRgb)
            count+=1
            frameRgb = None
            frameDepth = None

        if cv2.waitKey(1) == ord('q'):
            break

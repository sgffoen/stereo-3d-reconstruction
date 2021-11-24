import cv2
import numpy as np
import depthai as dai
import open3d as o3d
import tempfile
import os
import json

"""
USER INPUT
"""
# write frames
writeFrames = True
# filter
medianFilter = "7x7"
# Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
# Otherwise (False), the aligned depth is automatically upscaled to 1080p
downscaleColor = True
fps = 10
# The disparity is computed at this resolution, then upscaled to RGB resolution
# Select the camera sensor resolution: 1280×720, 1280×800, 640×400 (THE_720_P, THE_800_P, THE_400_P)
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
"""
END USER INPUT
"""

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

medianMap = {
    "OFF": dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
    "3x3": dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
    "5x5": dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
    "7x7": dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
}
if medianFilter not in medianMap:
    exit("Unsupported median size!")

median = medianMap[medianFilter]

stereo.initialConfig.setConfidenceThreshold(200)
stereo.initialConfig.setMedianFilter(median)  # KERNEL_7x7 default
# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

# Linking
camRgb.isp.link(rgbOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.depth.link(depthOut.input) #stereo.disparity.link(depthOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    fd, path = tempfile.mkstemp(suffix='.json')
    with os.fdopen(fd, 'w') as tmp:
        json.dump({
            "width": 1280,
            "height": 720,
            "intrinsic_matrix": [item for row in device.get_right_intrinsic() for item in row]
        }, tmp)

    device.getOutputQueue(name="rgb",   maxSize=4, blocking=False)
    device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    frameRgb = None
    frameDepth = None
    frame_cnt=0
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
            #if 1: frameDepth = (frameDepth * 255. / maxDisparity).astype(np.uint8)
            # Optional, apply false colorization
            #if 1: frameDepth = cv2.applyColorMap(frameDepth, cv2.COLORMAP_HOT)
            frameDepth = np.ascontiguousarray(frameDepth)
            cv2.imshow("depth", frameDepth)


        # Blend when both received
        if frameRgb is not None and frameDepth is not None:
            # # Need to have both frames in BGR format before blending
            # if len(frameDepth.shape) < 3:
            #     frameDepth = cv2.cvtColor(frameDepth, cv2.COLOR_GRAY2BGR)
            # # TODO add a slider to adjust blending ratio
            # blended = cv2.addWeighted(frameRgb, 0.6, frameDepth, 0.4 ,0)
            # cv2.imshow("rgb-depth", blended)
            rgb = o3d.geometry.Image((frameRgb).astype(np.uint8))
            depth = o3d.geometry.Image((frameDepth).astype(np.uint16))
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)

            skip = 5*fps # skip first 5 seconds before writing frames
            if writeFrames is True and frame_cnt>skip:
                o3d.io.write_image("../dataset/depth/depth{}.png".format(str(frame_cnt-skip).zfill(5)), depth)
                o3d.io.write_image("../dataset/image/rgb{}.png".format(str(frame_cnt-skip).zfill(5)), rgb)
                #cv2.imwrite("data/depth/framedepth%d.png" % (frame_cnt-skip), frameDepth)
                #cv2.imwrite("data/image/framergb%d.png" % (frame_cnt-skip), frameRgb)
            frame_cnt+=1
            frameRgb = None
            frameDepth = None

        if cv2.waitKey(1) == ord('q'):
            break

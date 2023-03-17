import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs

# Create DepthAI pipeline
pipeline = depthai.Pipeline()

VIDEO_SIZE = (1072, 1072)
# Create ColorCamera node
cam_rgb = pipeline.create(depthai.node.ColorCamera)
cam_rgb.setPreviewSize(VIDEO_SIZE)
# ?
cam_rgb.setInterleaved(False)

# MODEL
detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
# Set path of the blob (NN model). We will use blobconverter to convert&download the model
# detection_nn.setBlobPath("/path/to/model.blob")
#detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
detection_nn.setBlobPath(blobconverter.from_zoo(name='face-recognition-resnet100-arcface-onnx', shaves=6))
detection_nn.setConfidenceThreshold(0.5)

cam_rgb.preview.link(detection_nn.input)

xout_rgb = pipeline.create(depthai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.create(depthai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


with depthai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")

    frame = None
    detections = []
    while True:

        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            detections = in_nn.detections
        if frame is not None:
            for detection in detections:

                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                print(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord('q'):
            break







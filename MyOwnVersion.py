# Import dependencies
import glob
import shutil

import cv2
import depthai
import os
import argparse
import blobconverter
import numpy as np
from MultiMsgSync import TwoStageHostSeqSync

PATH_WHERE_DATA_SAVED = "face_to_detect"

class TextHelper:
    # Helper class to add text to images with OpenCV
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 4, self.line_type)  #create a thicker outline
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 2, self.line_type)  #create the actual text.


class FaceRecognition:
    """
    Perform face recognition using cosine similarity.
    Initialized with a path to the database of known faces and a name for the face being recognized.
    """
    def __init__(self, db_path, name) -> None:
        self.read_db(db_path)
        self.name = name
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.printed = True

    def cosine_distance(self, a, b):
        """
        Calculates the cosine distance between two feature vectors.
        """
        if a.shape != b.shape:
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b.T) / (a_norm * b_norm)

    def new_recognition(self, results):
        """
        Performs face recognition on a given feature vector, by calculating the cosine distance between the given
         feature vector and all known faces in the database, and selecting the label with the highest similarity score.

        If the similarity score is below a certain threshold (0.5), the face is labeled as "UNKNOWN".
        """
        conf = []
        max_ = 0
        label_ = None
        for label in list(self.labels):
            for j in self.db_dic.get(label):
                conf_ = self.cosine_distance(j, results)
                if conf_ > max_:
                    max_ = conf_
                    label_ = label

        conf.append((max_, label_))
        name = conf[0] if conf[0][0] >= 0.5 else (1 - conf[0][0], "UNKNOWN")
        # self.putText(frame, f"name:{name[1]}", (coords[0], coords[1] - 35))
        # self.putText(frame, f"conf:{name[0] * 100:.2f}%", (coords[0], coords[1] - 10))

        # name format example : (1, 'UNKNOWN')
        # If face is not known yet
        if name[1] == "UNKNOWN":
            self.create_db(results)
        return name

    def read_db(self, databases_path):
        """
        Read the face database from the specified path and stores it in a dictionary with labels as keys and feature vectors as values.
        """
        self.labels = []
        for file in os.listdir(databases_path):
            filename = os.path.splitext(file)
            if filename[1] == ".npz":
                self.labels.append(filename[0])

        self.db_dic = {}
        for label in self.labels:
            with np.load(f"{databases_path}/{label}.npz") as db:
                self.db_dic[label] = [db[j] for j in db.files]

    def putText(self, frame, text, coords):
        """
        Add text to an image.
        """
        cv2.putText(frame, text, coords, self.text_type, 1, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1, self.color, 1, self.line_type)

    def create_db(self, results):
        """
        Save the feature vector of a new face to the database, using the specified name.
        If no name is specified, a message is printed to the console.
        """
        if self.name is None:
            if not self.printed:
                print("Wanted to create new DB for this face, but --name wasn't specified")
                self.printed = True
            return
        print('Saving face...')
        try:
            saved_files = glob.glob(PATH_WHERE_DATA_SAVED + '/*')
            # Delete old files
            for f in saved_files:
                os.remove(f)

            with np.load(f"{PATH_WHERE_DATA_SAVED}/{self.name}.npz") as db:
                db_ = [db[j] for j in db.files][:]
        except Exception as e:
            db_ = []
        db_.append(np.array(results))
        np.savez_compressed(f"{PATH_WHERE_DATA_SAVED}/{self.name}", *db_)
        self.adding_new = False

# Initialization
""" 
If We press the Face recognition button : 
If there is an arg => We can recognize the face
else: 
Error box message
"""
"""
Usage : python filename.py -n your_name or --name"""
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="Name of the person for database saving")
args = argParser.parse_args()

if args.name:
    # Check if the folder already exists
    if os.path.exists(PATH_WHERE_DATA_SAVED):
        # Remove it and all its contents since the user wants to recognize a new face
        print(f"Deleting existing folder: {PATH_WHERE_DATA_SAVED}")
        shutil.rmtree(PATH_WHERE_DATA_SAVED)
    # Create the folder
    os.makedirs(PATH_WHERE_DATA_SAVED)
else:
    # get all files with the given extension in the folder
    npz_file = glob.glob(os.path.join(PATH_WHERE_DATA_SAVED, "*" + "npz"))
    if len(npz_file) <1:
        # Can't recognize a face so we exit / Or we follow the Unknown person ?
        exit(0)

VIDEO_SIZE = (1072, 1072)

# Create General DepthAI pipeline - Will allow to create a sequence of nodes that process our camera frames (input) and generate output data, such as object detections.
pipeline = depthai.Pipeline()

# 1 - Create Color Camera node
cam_rgb = pipeline.create(depthai.node.ColorCamera)  # Or use .createColorCamera() method
cam_rgb.setPreviewSize(VIDEO_SIZE)  # Set the size of the preview camera box (interface)
cam_rgb.setVideoSize(VIDEO_SIZE)  # Modify the input frame size (ImageManip rotate requires a multiple of 16)
# Specify which board socket to use - OAK-1 selects automatically the RGB one, so we do not need to specify it
# cam.setBoardSocket(dai.CameraBoardSocket.RGB)
# Same for cam resolution, which is already 1080p on OAK1
#cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)  #
cam_rgb.setInterleaved(False)  # In non-interleaved mode, each pixel is output separately, which can make it easier to process the camera data.

# 2 - Create XLinkOut node
# Used to send data from the device (OAK Camera preview) to the host via XLink, so the host can then display it
xout_rgb = pipeline.create(depthai.node.XLinkOut)
xout_rgb.setStreamName("rgb")  # Tag the output data of the XLinkOut node - Make sure to don't change this name
cam_rgb.preview.link(xout_rgb.input)  # Connects the camera output to the XLinkOut  node input

# 3 - Create 2 ImageManip nodes
# ImageManip node can be used to crop, rotate rectangle area or perform various image transforms: rotate, mirror, flip, perspective transform.

# 3.1 - Create an ImageManip node to have more frames in the pool.
"""
We create copies of the camera output frames, so that more frames can be stored in the pipeline buffer.
This is necessary because the preview output camera node can only have 4 frames in the pool before it will wait (freeze). 
By copying frames and setting the ImageManip pool size to a higher number, this issue can be fixed. 
The setNumFramesPool() method sets the number of frames in the pool
The setMaxOutputFrameSize() method sets the maximum output frame size in bytes.
"""

copy_manip = pipeline.create(depthai.node.ImageManip)
cam_rgb.preview.link(copy_manip.inputImage)
copy_manip.setNumFramesPool(20)
copy_manip.setMaxOutputFrameSize(1072*1072*3)

# 5 - Create an ImageManip node that takes the previous one as its input and that will crop the frame before sending it to the face detection NN model (required size for processing by the NN node).
face_det_manip = pipeline.create(depthai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300)
copy_manip.out.link(face_det_manip.inputImage)

# 6 - Create a MobileNet-based object detection Neural Network node
face_det_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
face_det_nn.setConfidenceThreshold(0.5)  # Confidence threshold for face detection (minimum score a detected object must have to be considered a valid detection)
face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))  # Set path of the blob (binary) file containing the pre-trained face detection model. We will use blobconverter to convert & download the model from OpenVINO Zoo. shaves parameter sets the number of NCEs (Neural Compute Engines) to be used by the neural network accelerator node. A higher number of shaves means more parallel processing power, but also higher power consumption and heat generation. In this case, 6 shaves are used, which is suitable for the OAK-1 device.

# Link Face Face detection NN node input to the 2nd ImageManip node output
face_det_manip.out.link(face_det_nn.input)

# 7 - Send data (NN Result) from the device to the host
face_det_xout = pipeline.create(depthai.node.XLinkOut)
face_det_xout.setStreamName("detection")
face_det_nn.out.link(face_det_xout.input)

# 8 - Create a Script node to run custom Python scripts on the device (camera)
"""
It will take the output from the face detection NN as an input and set ImageManipConfig
to the 'age_gender_manip' to crop the initial frame
"""
script = pipeline.create(depthai.node.Script)
script.setProcessor(depthai.ProcessorType.LEON_CSS)  # LEON is low-power processor used by the OAK-1

face_det_nn.out.link(script.inputs['face_det_in'])  # Connect the output of face_det_nn to the input named face_det_in of the Script node, to pass the detected faces to the script

# We also interested in sequence number for syncing
face_det_nn.passthrough.link(script.inputs['face_pass'])  # Connects the output of the face_det_nn node to the input named face_pass of the Script node, to pass through the sequence number of each detected face, which can be used for synchronizing the face detection results with other nodes in the pipeline.

# Connects the output of the copy_manip node to the input named preview of the Script node, to allow the processed camera output frames to be passed to script, so that they can be further processed or displayed.
copy_manip.out.link(script.inputs['preview'])

# Read Script
with open("script.py", "r") as f:
    script.setScript(f.read())

print("Creating Head pose estimation NN")

# 9 - Create a 3rd Image Manip Node to resize images to 60x60
headpose_manip = pipeline.create(depthai.node.ImageManip)
headpose_manip.initialConfig.setResize(60, 60)
headpose_manip.setWaitForConfigInput(True)
# Send manip_cfg & manip_img script outputs to the input of the headpose_manip node for further processing.
script.outputs['manip_cfg'].link(headpose_manip.inputConfig)
script.outputs['manip_img'].link(headpose_manip.inputImage)

# 10 - Create a new Neural Network node for head pose estimation
headpose_nn = pipeline.create(depthai.node.NeuralNetwork)
headpose_nn.setBlobPath(blobconverter.from_zoo(name="head-pose-estimation-adas-0001", shaves=6))
headpose_manip.out.link(headpose_nn.input)

# Redirect script outputs
headpose_nn.out.link(script.inputs['headpose_in'])
headpose_nn.passthrough.link(script.inputs['headpose_pass'])

print("Creating face recognition ImageManip/NN")

# 11 - Create a 4 th Image Manip Node to resize images to 112x112
face_rec_manip = pipeline.create(depthai.node.ImageManip)
face_rec_manip.initialConfig.setResize(112, 112)
face_rec_manip.inputConfig.setWaitForMessage(True)  # sets the node to wait for configuration messages to arrive before processing any frames.

# Link so configuration messages and frames sent by script node can be received by the face_rec_manip node for processing.
script.outputs['manip2_cfg'].link(face_rec_manip.inputConfig)
script.outputs['manip2_img'].link(face_rec_manip.inputImage)

# 12 - Create a new Neural Network node for face recognition
face_rec_nn = pipeline.create(depthai.node.NeuralNetwork)
face_rec_nn.setBlobPath(blobconverter.from_zoo(name="face-recognition-arcface-112x112", zoo_type="depthai", shaves=6))
face_rec_manip.out.link(face_rec_nn.input)

# send data from the device to the host
arc_xout = pipeline.create(depthai.node.XLinkOut)
arc_xout.setStreamName('recognition')
face_rec_nn.out.link(arc_xout.input)

def frame_norm(frame, bbox):
    """
    Converts this normalized bbox to pixel coordinates that are relative to the size of the frame.
    :param frame: Image in normalized form
    :param bbox: Bounding Box in normalized form
    :return:
    """
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


with depthai.Device(pipeline) as device:
    facerec = FaceRecognition(PATH_WHERE_DATA_SAVED, args.name)
    sync = TwoStageHostSeqSync()
    text = TextHelper()

    queues = {}
    # Create output queues
    for name in ["rgb", "detection", "recognition"]:
        queues[name] = device.getOutputQueue(name)
        # print("Queues are:", queues, queues[name])

    while True:
        for name, q in queues.items():
            # Add all msgs (color frames, object detections and face recognitions) to the Sync class.
            if q.has():
                sync.add_msg(q.get(), name)

        msgs = sync.get_msgs()
        if msgs is not None:
            frame = msgs["rgb"].getCvFrame()
            #frame = np.flipud(frame)
            dets = msgs["detection"].detections
            # print("frame:", frame, "dets:", dets)

            for i, detection in enumerate(dets):
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                #print(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2) #color, thickness
                # cv2.rectangle(frame, (0.4, 0.4), (0.6, 0.6), (10, 245, 10), 2) #color, thickness

                features = np.array(msgs["recognition"][i].getFirstLayerFp16())
                # print("features", features)
                conf, name = facerec.new_recognition(features)
                # print("conf", conf, "name", name)
                text.putText(frame, f"{name} {(100*conf):.0f}%", (bbox[0] + 10,bbox[1] + 35))

            cv2.imshow("rgb", cv2.resize(frame, (800,800)))

        if cv2.waitKey(1) == ord('q'):
            break

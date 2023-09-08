import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Define a class for fall detection
class FallDetection:
    def __init__(self):
        # Determine whether to use GPU (CUDA) if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load a pre-trained Keypoint RCNN model with the KEYPOINTRCNN_RESNET50_FPN backbone
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, skeleton_cache):
        '''
        This function takes a cache of skeletons as input, with a shape of (M x 17 x 2),
        where M represents the number of skeletons (time steps).
        The number 17 represents the count of points in each skeleton (17 key points),
        and 2 represents the (x, y) coordinates.

        This function uses the cache to detect falls.

        The function will return:
            - bool: isFall (True or False)
            - float: fallScore
        '''
        # If the input is a string, load the skeleton_cache as a NumPy array
        if isinstance(skeleton_cache, str):
            skeleton_cache = np.load(skeleton_cache)

        # Check for and handle missing or invalid points in the skeleton_cache
        skeleton_cache = self.handle_missing_points(skeleton_cache)

        # Perform fall detection logic here
        isFall, fallScore = self.detect_fall(skeleton_cache)

        return isFall, fallScore

    # Method to handle missing or invalid points in the skeleton data
    def handle_missing_points(self, skeleton_cache):
        # Handle missing points (NaN or <= 0 values) in the skeleton cache
        skeleton_cache[np.isnan(skeleton_cache)] = 0

        return skeleton_cache

    # Method to calculate the angle between two vectors defined by keypoints
    def calculate_angle(self, keypoint1, keypoint2, keypoint3):
        vector1 = keypoint1 - keypoint2
        vector2 = keypoint3 - keypoint2
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        cosine_angle = dot_product / (norm1 * norm2)
        angle = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle)
        return angle_degrees

    # Method to detect falls in the skeleton data
    def detect_fall(self, skeleton_cache):
        # Initialize variables for fall detection
        frame_count = 0
        angle_sum_list = []

        isFall = False
        fallScore = 500

        # Get the number of frames (time steps) in the skeleton_cache
        M = skeleton_cache.shape[0]

        # Iterate through frames in the skeleton_cache
        for frame_data in skeleton_cache:
            if frame_data is not None:
                total_angle = 0.0

                # Perform inference with Keypoint RCNN on the frame
                outputs = self.predictor(frame_data)

                # Get the keypoints from the outputs
                keypoints = outputs["instances"].pred_keypoints.cpu().numpy()

                # Define the keypoint pairs for angle calculation
                joint_pairs = [
                    ("nose", "left_shoulder"),    # Nose to left shoulder
                    ("nose", "right_shoulder"),   # Nose to right shoulder
                    ("left_eye", "left_shoulder"),  # Left eye to left shoulder
                    ("right_eye", "right_shoulder"),  # Right eye to right shoulder
                    ("left_ear", "left_shoulder"),  # Left ear to left shoulder
                    ("right_ear", "right_shoulder"),  # Right ear to right shoulder
                    ("left_shoulder", "left_elbow"),  # Left shoulder to left elbow
                    ("right_shoulder", "right_elbow"),  # Right shoulder to right elbow
                    ("left_elbow", "left_wrist"),  # Left elbow to left wrist
                    ("right_elbow", "right_wrist"),  # Right elbow to right wrist
                    ("left_shoulder", "left_hip"),  # Left shoulder to left hip
                    ("right_shoulder", "right_hip"),  # Right shoulder to right hip
                    ("left_hip", "left_knee"),  # Left hip to left knee
                    ("right_hip", "right_knee"),  # Right hip to right knee
                    ("left_knee", "left_ankle"),  # Left knee to left ankle
                    ("right_knee", "right_ankle"),  # Right knee to right ankle
                ]

                # Calculate angles between specified keypoint pairs for the current frame
                for joint_pair in joint_pairs:
                    joint1, joint2 = joint_pair
                    index1 = outputs["instances"].get_fields()["pred_keypoints"].get_field(joint1).cpu().numpy()
                    index2 = outputs["instances"].get_fields()["pred_keypoints"].get_field(joint2).cpu().numpy()
                    
                    # Calculate the angle using the indices of the keypoints
                    keypoint1 = keypoints[index1[0]][:2]
                    keypoint2 = keypoints[index2[0]][:2]
                    
                    angle = self.calculate_angle(keypoint1, keypoint2, keypoint2)

                    total_angle += angle

                # Append the total angle for the current frame to the list
                angle_sum_list.append(total_angle)

                # Increment the frame count
                frame_count += 1

                # Calculate the differences between consecutive angle sums
                if frame_count >= M:
                    angle_differences = [angle_sum_list[i] - angle_sum_list[i - 1] for i in range(1, M)]

                    # Calculate the average of the angle differences
                    average_difference = sum(angle_differences) / len(angle_differences)

                    # Check if the average difference exceeds the fallScore threshold
                    if average_difference > M * fallScore:
                        print("Fall Detected!")
                        isFall = True
                        break

                    # Reset frame_count and angle_sum_list for the next set of frames
                    frame_count = 0
                    angle_sum_list.clear()

        return isFall, fallScore




import cv2
import mediapipe as mp
import numpy as np



def ky_lor(input_dir):
    keypoints, mask = get_pose(input_dir)
    lor_angle, lor_label = lordosis_angle(keypoints, mask, thresholds=[30, 40, 50])
    kyph_angle, kyph_label = kyphosis_angle(keypoints, mask, thresholds=[40, 50, 60])
    out_dict = {"lordosis": lor_label, "kyphosis": kyph_label}
    return out_dict


def get_pose(image, tightness=0.65):
    mp_pose = mp.solutions.pose
    # Create a MediaPipe `Pose` object
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                      model_complexity=2,
                      enable_segmentation=True) as pose:
        # Read the file in and get dims
        image = cv2.imread(image)

        # Convert the BGR image to RGB and then process with the `Pose` object.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Extract landmarks
    landmarks = results.pose_landmarks.landmark
    keypoints = dict()
    keypoints['mouth_r'] = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,
                            landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
    keypoints['elbow_l'] = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    keypoints['hip_l'] = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    keypoints['shoulder_l'] = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    mask = np.where(results.segmentation_mask > tightness, 1, 0)
    return keypoints, mask


def kyphosis_angle(keypoints, mask, thresholds, l1=30, h1=25):
    """
        kyphosis_angle calculates the angle for kyphosis and label it.

        :param keypoints: human keypoints(joints)
        :param mask: segmentation mask of human
        :param thresholds: kyphosis thresholds for labeling
        :param l1: hyperparameter for the cut of image (higher number cause bigger cut from upside)
        :param h1: hyperparameter for the cut of image (higher number cause shorter cut from below)
        :return: return the angle and label for the risk of kyphosis
        """
    mouth = keypoints['mouth_r']
    elbow = keypoints['elbow_l']
    shoul = keypoints['shoulder_l']
    shoul_x = int(shoul[0] * mask.shape[1])
    shoul_y = int(shoul[1] * mask.shape[0])
    el_x = int(elbow[0] * mask.shape[1])
    el_y = int(elbow[1] * mask.shape[0])
    relative_x = int(mouth[0] * mask.shape[1])
    relative_y = int(mouth[1] * mask.shape[0])
    x, y = relative_x, relative_y
    while mask[y, x]:
        y += 1
    y -= 1

    while mask[y, x]:
        x -= 1
    x += 1
    piece = mask[shoul_y - l1:el_y - h1, :shoul_x]  # y+10:el_y-50, :x+10]
    smooth = cv2.GaussianBlur(np.uint8(piece), (0, 0), sigmaX=0.5, sigmaY=0.5, borderType=cv2.BORDER_DEFAULT)
    edges = cv2.Canny(smooth, 0, 1)
    a = np.where(edges == 255)
    p1 = np.array((a[1][0], a[0][0]))
    p2 = np.array((a[1].min(), a[0][a[1].argmin()]))
    p3 = np.array((a[1][a[1].argmin():].max(), a[0][a[1][a[1].argmin():].argmax() + a[1].argmin()]))

    L = np.linalg.norm(p3 - p1)
    H = np.linalg.norm(np.cross(p3 - p1, p1 - p2)) / L
    theta = 4 * np.arctan(2 * np.abs(H) / np.abs(L)) * 180 / np.pi
    label = label_maker(theta, thresholds)
    return theta, label


def lordosis_angle(keypoints, mask, thresholds, l1=40, h1=50):
    """
    lordosis_angle calculates the angle for lordosis and label it.

    :param keypoints: human keypoints(joints)
    :param mask: segmentation mask of human
    :param thresholds: lordosis thresholds for labeling
    :param l1: hyperparameter for the cut of image (higher number cause bigger cut from upside)
    :param h1: hyperparameter for the cut of image (higher number cause shorter cut from below)
    :return: return the angle and label for the risk of lordosis
    """
    elbow = keypoints['elbow_l']
    hip = keypoints['hip_l']
    el_x = int(elbow[0] * mask.shape[1])
    el_y = int(elbow[1] * mask.shape[0])
    hip_x, hip_y = int(hip[0] * mask.shape[1]), int(hip[1] * mask.shape[0])
    piece = mask[el_y - l1:hip_y - h1, :hip_x]
    smooth = cv2.GaussianBlur(np.uint8(piece), (0, 0), sigmaX=0.5, sigmaY=0.5, borderType=cv2.BORDER_DEFAULT)
    edges = cv2.Canny(smooth, 0, 1)
    a = np.where(edges == 255)
    p1 = np.array((a[1][0], a[0][0]))
    p2 = np.array((a[1].max(), a[0][a[1].argmax()]))
    p3 = np.array((a[1][a[1].argmax():].min(), a[0][a[1][a[1].argmax():].argmin() + a[1].argmax()]))
    L = np.linalg.norm(p3 - p1)
    H = np.linalg.norm(np.cross(p3 - p1, p1 - p2)) / L
    theta = 4 * np.arctan(2 * np.abs(H) / np.abs(L)) * 180 / np.pi
    label = label_maker(theta, thresholds)
    return theta, label


def label_maker(angle, thresholds):
    labels = [0.5, 1.5, 2.5, 3.5]
    for i, threshold in enumerate(thresholds):
        if angle <= threshold:
            return labels[i]
        if threshold == thresholds[-1]:
            return labels[-1]

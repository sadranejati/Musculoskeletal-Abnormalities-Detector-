# -*- coding: utf-8 -*-
"""
Created on Tue May  7 22:32:21 2024

@author: kian.imani
"""
def current_id():
  return(7)

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import time
import pandas as pd
import os

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle

def calculate_knee_distance(landmarks, mp_pose):
    left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
    right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])

    distance = np.linalg.norm(left_knee - right_knee)

    return distance

def calculate_pelvic_slope(landmarks, mp_pose):
    # Get the positions of the pelvic markers
    marker1 = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
    marker2 = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y])

    # Calculate the slope between the pelvic markers
    if (marker2[0] - marker1[0]) != 0:
        slope = (marker2[1] - marker1[1]) / (marker2[0] - marker1[0])
    else:
        slope = float('inf')

    return slope

def calculate_slope(point1, point2):
    # Calculate slope between two points
    if (point2.x - point1.x) != 0:
        return (point2.y - point1.y) / (point2.x - point1.x)
    else:
        return float('inf')

def forward_head_slope(landmarks, mp_pose):
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

    # Extract x and y coordinates
    right_shoulder_x = right_shoulder.x
    right_shoulder_y = right_shoulder.y
    right_ear_x = right_ear.x
    right_ear_y = right_ear.y

    vertical_line = np.array([right_shoulder_x, 1])  # Vertical line at the right shoulder

    slope = calculate_slope(right_shoulder, right_ear)

    return slope


def calculate_eyes_slope(landmarks, mp_pose):
    # Get the positions of the left and right eye markers
    left_eye = np.array([landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER].y])
    right_eye = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER].y])

    # Calculate the slope between the eyes markers
    if (right_eye[0] - left_eye[0]) != 0:
        slope = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])
    else:
        slope = float('inf')

    return slope




def calculate_ankle_distance(landmarks, mp_pose):
    left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
    right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])

    distance = np.linalg.norm(left_ankle - right_ankle)

    return distance

std_values = {
    "Zarbdari": (3.668312705+3.07996)/2,
    #"right knee angle": 3.07996,
    "Parantezi": 0.02293,
    "PelvicSlope": 0.01304,
    "EyesSlope": 0.02897,
    "ShoulderSlope": 0.01707,
    "ForwardHeadSlope":8.13029

}

avg_values = {
    #"Left Ankle": 25,
    #"Right Ankle": 30,
    #"Left Hip": 20,
    #"Right Hip": 25,
    "Leftkneeangle": 175.909,
    "rightkneeangle": 175.616,
    "Parantezi": 0.12024,
    "PelvicSlope": 0.01308,
    "EyesSlope": 0.02273,
    "ShoulderSlope": 0.01361,
    "Zarbdari":(175.909+175.616)/2,
    "ForwardHeadSlope":6.90848
}

def analyze_value(value, avg, std):
    value = abs(value)
    if  value < (avg + 1* std) and value > (avg + 0.5*std):
        return "کم"
    elif  value < (avg + 0.5* std):
        return "نرمال"
    elif  value > (avg + 2* std):
        return "زیاد"
    elif value < (avg + 2*std) and value > (avg + std):
        return "متوسط"
    else:
        return "نرمال"



def analyze_images(front_image_path, side_image_path, output_folder):
    front_results, side_results = {}, {}

    # Front Image Analysis
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        front_frame = cv2.imread(front_image_path)
        front_frame_rgb = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
        front_results = pose.process(front_frame_rgb)
        front_landmarks = front_results.pose_landmarks

        if front_landmarks:
            # Draw landmarks on the image
            annotated_image = front_frame.copy()
            mp_drawing.draw_landmarks(annotated_image, front_landmarks, mp_pose.POSE_CONNECTIONS)

            front_output_path = os.path.join(output_folder, str(current_id())+"_front_result.jpg")
            cv2.imwrite(front_output_path, annotated_image)

            front_pelvic_slope = calculate_pelvic_slope(front_landmarks.landmark, mp_pose)
            front_eyes_slope = calculate_eyes_slope(front_landmarks.landmark, mp_pose)
            front_shoulder_slope = calculate_slope(
                front_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                front_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            )
            front_knee_distance = calculate_knee_distance(front_landmarks.landmark, mp_pose)
            front_left_knee_angle = calculate_angle(
                [front_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, front_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                [front_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x, front_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                [front_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, front_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            )
            front_right_knee_angle = calculate_angle(
                [front_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, front_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                [front_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, front_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                [front_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, front_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            )

    # Side Image Analysis
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        side_frame = cv2.imread(side_image_path)
        side_frame_rgb = cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB)
        side_results = pose.process(side_frame_rgb)
        side_landmarks = side_results.pose_landmarks

        if side_landmarks:
            # Draw landmarks on the image
            annotated_image = side_frame.copy()
            mp_drawing.draw_landmarks(annotated_image, side_landmarks, mp_pose.POSE_CONNECTIONS)

            side_output_path = os.path.join(output_folder, str(current_id())+"_side_result.jpg")
            cv2.imwrite(side_output_path, annotated_image)

            side_forward_head_slope = forward_head_slope(side_landmarks.landmark, mp_pose)

    front_results_dict = {
        "PelvicSlope": front_pelvic_slope,
        "EyesSlope": front_eyes_slope,
        "ShoulderSlope": front_shoulder_slope,
        "Parantezi": (front_knee_distance/0.12024),
        "Zarbdari": (front_left_knee_angle+front_right_knee_angle)/(175.909+175.616),
        "RightKneeAngle": front_right_knee_angle
    }

    side_results_dict = {
        "ForwardHeadSlope": side_forward_head_slope
    }
    # Combine results
    combined_results = {}
    for key in front_results_dict:
        combined_results[key] = front_results_dict[key]
    for key in side_results_dict:
        combined_results[key] = side_results_dict[key]

    appended_dict = {}
    for key in combined_results:
        if key in avg_values:
            avg = avg_values[key]
            std = std_values[key]
            appended_dict[key + "Analysis"] = analyze_value(combined_results[key], avg, std)
        combined_results[key] = float(round(combined_results[key],4))

    return(combined_results | appended_dict)




front_image_path = "front.jpg"
side_image_path = "side.jpg"
output_folder = os.getcwd()+ "/analyzed/"

combined_results= analyze_images(front_image_path, side_image_path, output_folder)

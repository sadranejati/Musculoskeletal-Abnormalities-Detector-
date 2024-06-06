import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(0)
import tkinter as tk 
from tkinter import ttk 
from tkinter import filedialog as fd
import sqlite3
from PIL import ImageTk, Image, ImageDraw
import os 
import shutil 
import pathlib
import plotly.graph_objects as go
import glob
from numpy import cos, sin, radians
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import time
import pandas as pd
from pyautogui import screenshot

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

translateDict = {
    'PelvicSlope':'نابرابری لگن',
    'EyesSlope':'کج گردنی',
    'ShoulderSlope':'نابرابری شانه',
    'Parantezi':'زانوی پرانتزی',
    'Zarbdari':'زانوی ضربدری',
    'ForwardHeadSlope':'سر به جلو'
}

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




con = sqlite3.connect("database9.db")
cur = con.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS user(
id INTEGER PRIMARY KEY,
name TEXT,
place Text,
description TEXT,
phone TEXT,
sidePhoto TEXT,
frontPhoto TEXT,
PelvicSlope float,
EyesSlope float,
ShoulderSlope float,
Parantezi float,
Zarbdari float,
RightKneeAngle float,
ForwardHeadSlope float,
PelvicSlopeAnalysis TEXT,
EyesSlopeAnalysis TEXT,
ShoulderSlopeAnalysis TEXT,
ParanteziAnalysis TEXT,
ZarbdariAnalysis TEXT,
ForwardHeadSlopeAnalysis TEXT,
Lordosis float,
Kyphosis float)
""")

pathlib.Path(os.getcwd() + '/photos').mkdir(parents=True, exist_ok=True) 
pathlib.Path(os.getcwd() + '/analyzed').mkdir(parents=True, exist_ok=True) 

def current_id():
    cur.execute('select max(id) from user')
    last_id = cur.fetchone()[0]
    if last_id == None:
        return(1)
    else:
        return(int(last_id)+1)
def titleBar(window,home=False):
    titleBar = tk.Frame(window,background='#EF7A52')

    titleBar.place(relx=0,rely=0,height=50,relwidth=1)
    appTitle = tk.Label(titleBar,text='نرم افزار تشخیص ناهنجاری اسکلتی', font=('IRANSans Bold', 12),background='#EF7A52',foreground='white')
    appTitle.pack(side='right',padx=5)
    if home:
        appExitButton = tk.Button(titleBar,text='خروج',font=('IRANSans',10),command=lambda:[window.quit(),updateTable()],highlightbackground='#EF7A52',activebackground='#EF7A52', foreground='#EF7A52',padx=10,borderwidth=0)
    else:
        appExitButton = tk.Button(titleBar,text='صفحه اصلی',font=('IRANSans',10),command=lambda:[window.destroy(),updateTable()],highlightbackground='#EF7A52',activebackground='#EF7A52', foreground='#EF7A52',padx=10,borderwidth=0)

    appExitButton.pack(side='left',padx=5)
def getUserByID(id):
    cur = con.cursor()
    cur.execute('select * from user where id ='+str(id))
    userDataFetched = cur.fetchall()[0]
    userData = {}
    userData['id'] = userDataFetched[0]
    userData['name'] = userDataFetched[1]
    userData['place'] = userDataFetched[2]
    userData['description'] = userDataFetched[3]
    userData['phone'] = userDataFetched[4]
    userData['sidephoto'] = userDataFetched[5]
    userData['frontphoto'] = userDataFetched[6]
    userData['PelvicSlope'] = userDataFetched[7]
    userData['EyesSlope'] = userDataFetched[8]
    userData['ShoulderSlope'] = userDataFetched[9]
    userData['Parantezi'] = userDataFetched[10]
    userData['Zarbdari'] = userDataFetched[11]
    userData['RightKneeAngle'] = userDataFetched[12]
    userData['ForwardHeadSlope'] = userDataFetched[13]
    userData['PelvicSlopeAnalysis'] = userDataFetched[14]
    userData['EyesSlopeAnalysis'] = userDataFetched[15]
    userData['ShoulderSlopeAnalysis'] = userDataFetched[16]
    userData['ParanteziAnalysis'] = userDataFetched[17]
    userData['ZarbdariAnalysis'] = userDataFetched[18]
    userData['ForwardHeadSlopeAnalysis'] = userDataFetched[19]
    userData['Lordosis'] = userDataFetched[20]
    userData['Kyphosis'] = userDataFetched[21]
    return(userData)
def updateTable(search=None):   
    global table 
    global cur
    for row in table.get_children():
        table.delete(row)
    if search==None:
        cur.execute('select * from user order by id desc limit 100')
        global searchNameVar
        searchNameVar.set('')
    else:
        cur.execute("select * from user where name like '%"+search+"%' or place like '%"+search+"%' or description like '%"+search+"%' or phone like '%"+search+"%' or id like '%"+search+"%' order by id desc limit 100")
    dataset = cur.fetchall()
    for i in dataset[::-1]:
        data = (i[0], i[1], i[2], i[4])[::-1]
        table.insert(parent = '', index = 0, values = data)

def analyzeCase(sidePhotoAdress, frontPhotoAdress):
    sidePhotoEditedAdress = None 
    frontPhotoEditedAdress = None
    AnalyzedParameters = {}
    return sidePhotoEditedAdress, frontPhotoEditedAdress, AnalyzedParameters

def zoomImage(imageAddress):
    zoomImageWindow = tk.Toplevel()
    zoomImageWindow.title('بزرگ نمایی تصویر')
    global imgZoomed
    img = Image.open(imageAddress)
    img.thumbnail((zoomImageWindow.winfo_screenwidth()*0.7,zoomImageWindow.winfo_screenwidth()*0.7))
    imgZoomed = ImageTk.PhotoImage(img)
    panel = tk.Label(zoomImageWindow, image = imgZoomed)
    panel.pack(side = "bottom", fill = "both", expand = "yes")

def profileShowPage(id,backhome=False):
    profileShowWindow = tk.Toplevel()
    profileShowWindow.config(background='white')
    profileShowWindow.geometry('1000x700+'+str(int(profileShowWindow.winfo_screenwidth()/2-500))+'+'+str(int(profileShowWindow.winfo_screenheight()/2-350)))
    profileShowWindow.overrideredirect(True)
    userData=getUserByID(id)
    pathlib.Path(os.getcwd() + '/tmp').mkdir(parents=True, exist_ok=True) 

    titleBar = tk.Frame(profileShowWindow,background='#EF7A52')

    titleBar.place(relx=0,rely=0,height=50,relwidth=1)
    appTitle = tk.Label(titleBar,text='نرم افزار تشخیص ناهنجاری اسکلتی', font=('IRANSans Bold', 12),background='#EF7A52',foreground='white')
    appTitle.pack(side='right',padx=5)
    if backhome:
        appExitButton2 = tk.Button(titleBar,text='صفحه اصلی',font=('IRANSans',10),command=lambda:[profileShowWindow.destroy(),updateTable()],highlightbackground='#EF7A52',activebackground='#EF7A52', foreground='#EF7A52',padx=10,borderwidth=0)
        appExitButton2.pack(side='left',padx=5)
        appExitButton = tk.Button(titleBar,text='افزودن فرد جدید',font=('IRANSans',10),command=lambda:[profileShowWindow.destroy(), addCasePage()],highlightbackground='#EF7A52',activebackground='#EF7A52', foreground='#EF7A52',padx=10,borderwidth=0)
        appExitButton.pack(side='left',padx=5)
    else: 
        global addCaseWindow
        appExitButton2 = tk.Button(titleBar,text='صفحه اصلی',font=('IRANSans',10),command=lambda:[profileShowWindow.destroy(), addCaseWindow.destroy()],highlightbackground='#EF7A52',activebackground='#EF7A52', foreground='#EF7A52',padx=10,borderwidth=0)
        appExitButton2.pack(side='left',padx=5)
        appExitButton = tk.Button(titleBar,text='افزودن فرد جدید',font=('IRANSans',10),command=lambda:[profileShowWindow.destroy(), addCasePage()],highlightbackground='#EF7A52',activebackground='#EF7A52', foreground='#EF7A52',padx=10,borderwidth=0)
        appExitButton.pack(side='left',padx=5)

    editCaseFormFrame = tk.Frame(profileShowWindow,pady=5,padx=5)
    editCaseFormFrame.config(background='white')
    editCaseFormFrame.columnconfigure(0,weight=1)
    editCaseFormFrame.columnconfigure(1,weight=1)
    editCaseFormFrame.columnconfigure(2,weight=1)
    editCaseFormFrame.columnconfigure(3,weight=1)
    editCaseFormFrame.columnconfigure(4,weight=1)

    caseNameVar = tk.StringVar()
    caseNameVar.set(userData['name'])
    editCaseNameEntry = tk.Entry(editCaseFormFrame,textvariable=caseNameVar,justify='right',bg='#e6e6e6',font=('IRANSans',10),borderwidth=0)    
    editCaseNameEntry.grid(row=0, column=4, ipady=5,ipadx=5,sticky='nsew',padx=5,pady=5)

    casePlaceVar = tk.StringVar()
    casePlaceVar.set(userData['place'])
    editCasePlaceEntry = tk.Entry(editCaseFormFrame,textvariable=casePlaceVar,justify='right',bg='#e6e6e6',font=('IRANSans',10),borderwidth=0)    
    editCasePlaceEntry.grid(row=1, column=4, ipady=5,ipadx=5, sticky='nsew',padx=5,pady=5)

    caseTextVar = tk.StringVar()
    caseTextVar.set(userData['description'])
    editCasePlaceEntry = tk.Entry(editCaseFormFrame,textvariable=caseTextVar,justify='right',bg='#e6e6e6',font=('IRANSans',10),borderwidth=0)    
    editCasePlaceEntry.grid(row=2, column=4, ipady=5,ipadx=5, sticky='nsew',padx=5,pady=5)
    
    casePhoneVar = tk.StringVar()
    casePhoneVar.set(userData['phone'])
    editCasePhoneEntry = tk.Entry(editCaseFormFrame,textvariable=casePhoneVar,justify='right',bg='#e6e6e6',font=('IRANSans',10),borderwidth=0)    
    editCasePhoneEntry.grid(row=3, column=4, ipady=5,ipadx=5, sticky='nsew',padx=5,pady=5)
   
    for key in userData:
        if key in avg_values and key in std_values:
            analysis = (userData[key+"Analysis"])
            if analysis=='زیاد':
                fontColor = 'red' 
            elif analysis=='متوسط':
                fontColor = 'orange'
            elif analysis=='کم':
                fontColor = 'gold'
            else:
                fontColor = 'green'

            avg = (avg_values[key])
            std = (std_values[key])
            maxim = avg+3*std
            val = userData[key]
            if val<0:
                val = -val
            if val>maxim:
                val=maxim
            minim = 0
            fig = go.Figure(go.Indicator(
                mode = "gauge",
                value = abs(userData[key]),
                domain = {'x': [0,1], 'y': [0,1]},
                title = {'text': "("+translateDict[key]+f" ({analysis}",'font':{'size':50,'color':fontColor}},
                gauge = {'axis': {'range': [0, avg+3*std]},
                            'bar': {'color': 'rgba(0,0,0,0)','thickness':0.3},
                            'steps' : [
                                {'range': [0, avg+0.5*std], 'color': "green"},
                                {'range': [avg+0.5*std, avg+std], 'color': "yellow"},
                                {'range': [avg+std, avg+2*std], 'color': "orange"},
                                {'range': [avg+2*std, avg+3*std], 'color': "red"}],}))


            fig.update_layout(
                font={'color': "black", 'family': "IranSans",'size':1 if key=='Zarbdari' else 25},
                xaxis={'showgrid': False, 'showticklabels':False, 'range':[-1,1]},
                yaxis={'showgrid': False, 'showticklabels':False, 'range':[0,1]},
                plot_bgcolor='rgba(0,0,0,0)'
                )

            ## by setting the range of the layout, we are effectively adding a grid in the background
            ## and the radius of the gauge diagram is roughly 0.9 when the grid has a range of [-1,1]x[0,1]

            theta = 180 * (maxim-val) / (maxim - minim)
            r= 0.9
            x_head = r * cos(radians(theta))
            y_head = r * sin(radians(theta))

            fig.add_annotation(
                ax=0,
                ay=0,
                axref='x',
                ayref='y',
                x=x_head,
                y=y_head,
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=8
                )

            fig.write_image(os.getcwd()+"/tmp/"+key+".png")
    kaiphos = userData['Kyphosis']
    if kaiphos==0.5:
        analysis = 'نرمال'
        kaiphosColor = 'green'
    elif kaiphos==1.5:
        analysis= 'کم'
        kaiphosColor = 'yellow'
    elif kaiphos==2.5:
        analysis = 'متوسط'
        kaiphosColor = 'orange'
    else:
        analysis = 'شدید'
        kaiphosColor = 'red'
    maxim = 4
    val = min(max(kaiphos,0),maxim)
    minim = 0
    fig = go.Figure(go.Indicator(
        mode = "gauge",
        value = val,
        domain = {'x': [0,1], 'y': [0,1]},
        title = {'text': "("+'کایفوز'+f" ({analysis}",'font':{'size':50,'color':kaiphosColor}},
        gauge = {'axis': {'range': [0, 4]},
                    'bar': {'color': 'rgba(0,0,0,0)','thickness':0.3},
                    'steps' : [
                        {'range': [0, 1], 'color': "green"},
                        {'range': [1, 2], 'color': "yellow"},
                        {'range': [2, 3], 'color': "orange"},
                        {'range': [3, 4], 'color': "red"}],}))


    fig.update_layout(
        font={'color': "black", 'family': "IranSans",'size':1},
        xaxis={'showgrid': False, 'showticklabels':False, 'range':[-1,1]},
        yaxis={'showgrid': False, 'showticklabels':False, 'range':[0,1]},
        plot_bgcolor='rgba(0,0,0,0)'
        )

    ## by setting the range of the layout, we are effectively adding a grid in the background
    ## and the radius of the gauge diagram is roughly 0.9 when the grid has a range of [-1,1]x[0,1]

    theta = 180 * (maxim-val) / (maxim - minim)
    r= 0.9
    x_head = r * cos(radians(theta))
    y_head = r * sin(radians(theta))

    fig.add_annotation(
        ax=0,
        ay=0,
        axref='x',
        ayref='y',
        x=x_head,
        y=y_head,
        xref='x',
        yref='y',
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=10
        )
    fig.write_image(os.getcwd()+"/tmp/"+"kaiphos"+".png")

    lordosis = userData['Lordosis']
    if lordosis==0.5:
        analysis = 'نرمال'
        lordosisColor = 'green'
    elif lordosis==1.5:
        analysis= 'کم'
        lordosisColor = 'yellow'
    elif lordosis==2.5:
        analysis = 'متوسط'
        lordosisColor = 'orange'
    else:
        analysis = 'شدید'
        lordosisColor = 'red'
    maxim = 4
    val = min(max(lordosis,0),maxim)
    minim = 0
    fig = go.Figure(go.Indicator(
        mode = "gauge",
        value = val,
        domain = {'x': [0,1], 'y': [0,1]},
        title = {'text': "("+'لوردوز'+f" ({analysis}",'font':{'size':50,'color':lordosisColor}},
        gauge = {'axis': {'range': [0, 4]},
                    'bar': {'color': 'rgba(0,0,0,0)','thickness':0.3},
                    'steps' : [
                        {'range': [0, 1], 'color': "green"},
                        {'range': [1, 2], 'color': "yellow"},
                        {'range': [2, 3], 'color': "orange"},
                        {'range': [3, 4], 'color': "red"}],}))


    fig.update_layout(
        font={'color': "black", 'family': "IranSans",'size':1},
        xaxis={'showgrid': False, 'showticklabels':False, 'range':[-1,1]},
        yaxis={'showgrid': False, 'showticklabels':False, 'range':[0,1]},
        plot_bgcolor='rgba(0,0,0,0)'
        )


    ## by setting the range of the layout, we are effectively adding a grid in the background
    ## and the radius of the gauge diagram is roughly 0.9 when the grid has a range of [-1,1]x[0,1]

    theta = 180 * (maxim-val) / (maxim - minim)
    r= 0.9
    x_head = r * cos(radians(theta))
    y_head = r * sin(radians(theta))

    fig.add_annotation(
        ax=0,
        ay=0,
        axref='x',
        ayref='y',
        x=x_head,
        y=y_head,
        xref='x',
        yref='y',
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=10
        )
    fig.write_image(os.getcwd()+"/tmp/"+"lordosis"+".png")
    
    # guagesFrame = tk.Frame(profileShowWindow,pady=5,padx=5)
    # guagesFrame.config(background='black')
    # guagesFrame.columnconfigure(0,weight=1)
    # guagesFrame.columnconfigure(1,weight=1)
    # guagesFrame.columnconfigure(2,weight=1)
    # guagesFrame.rowconfigure(0,weight=1)
    # guagesFrame.rowconfigure(1,weight=1)
    # guagesFrame.place(width=1000,y=50)
    guageimgs = []
    guagePhotoimgs = []
    guageImgPanels = []
    for i in range(len((glob.glob(os.getcwd()+"/tmp/*.png")))):
        guageimgs.append(Image.open((glob.glob(os.getcwd()+"/tmp/*.png"))[i]))
        guageimgs[i].thumbnail((200,200))
        guagePhotoimgs.append(ImageTk.PhotoImage(guageimgs[i]))
        guageImgPanels.append(tk.Label(profileShowWindow, image = guagePhotoimgs[i],background='white'))
        guageImgPanels[i].image = guagePhotoimgs[i]
    guageImgPanels[0].bind(f"<Button-1>", lambda e:zoomImage((glob.glob(os.getcwd()+"/tmp/*.png"))[0]))
    guageImgPanels[1].bind(f"<Button-1>", lambda e:zoomImage((glob.glob(os.getcwd()+"/tmp/*.png"))[1]))
    guageImgPanels[2].bind(f"<Button-1>", lambda e:zoomImage((glob.glob(os.getcwd()+"/tmp/*.png"))[2]))
    guageImgPanels[3].bind(f"<Button-1>", lambda e:zoomImage((glob.glob(os.getcwd()+"/tmp/*.png"))[3]))
    guageImgPanels[4].bind(f"<Button-1>", lambda e:zoomImage((glob.glob(os.getcwd()+"/tmp/*.png"))[4]))
    guageImgPanels[5].bind(f"<Button-1>", lambda e:zoomImage((glob.glob(os.getcwd()+"/tmp/*.png"))[5]))
    guageImgPanels[6].bind(f"<Button-1>", lambda e:zoomImage((glob.glob(os.getcwd()+"/tmp/*.png"))[6]))
    guageImgPanels[7].bind(f"<Button-1>", lambda e:zoomImage((glob.glob(os.getcwd()+"/tmp/*.png"))[7]))
    for i in range(len((glob.glob(os.getcwd()+"/tmp/*.png")))):
        guageImgPanels[i].place(x=i*254+10 if i<4 else (i-4)*254+10, y=420 if i<4 else 560)


    # c= tk.Canvas(profileShowWindow,width=150, height=310,bg='white',bd=0, highlightthickness=0, relief='ridge')
    # c.place(x=820,y=380)
    # #Draw an Oval in the canvas
    # c.create_oval(0,0,150,150,fill='green',outline='white')
    # c.create_oval(0,160,150,310,fill='orange',outline='white')
    # lordosisText = tk.Label(c,text='لوردوز', font=('IRANSans Bold', 12),foreground='white',background='green')
    # lordosisText.place(x=56,y=58)
    # kyphosisText = tk.Label(c,text='کایفوز', font=('IRANSans Bold', 12),foreground='white',background='orange')
    # kyphosisText.place(x=56,y=221)
    # address = (glob.glob(os.getcwd()+"/tmp/*.png"))[0]
    # guageimg = Image.open(address)
    # guageimg.thumbnail((250,250))
    # guagePhotoimg = ImageTk.PhotoImage(guageimg)
    # guageImgPanel = tk.Label(profileShowWindow, image = guagePhotoimg,background='white')
    # guageImgPanel.image =  guagePhotoimg
    # guageImgPanel.bind("<Button-1>", lambda e:zoomImage(address))
    # guageImgPanel.place(x=10, y=360)

    def editCaseEdit(id, name, place, text, phone):
        cur = con.cursor()
        cur.execute("UPDATE user SET name = ?, place = ?, description = ?, phone = ? WHERE id = ?",(name,place,text,phone,id))
        con.commit() 
        updateTable()

    def editCaseDelete(id,backhome=False):
        cur = con.cursor()
        cur.execute("delete from user where id ="+str(id))
        con.commit()
        updateTable()
        profileShowWindow.destroy()
        if backhome==True:
            cur = con.cursor()
            cur.execute("SELECT count(*) FROM user WHERE id <"+str(userData['id']))
            if cur.fetchone()[0]!=0:
                cur.execute("SELECT max(id) FROM user WHERE id <"+str(userData['id']))
                lastid = cur.fetchone()[0]
                profileShowPage(lastid,backhome=True)  
            else:
                cur = con.cursor()
                cur.execute("SELECT count(*) FROM user WHERE id >"+str(userData['id']))
                if cur.fetchone()[0]!=0:
                    cur.execute("SELECT min(id) FROM user WHERE id >"+str(userData['id']))
                    nextid = cur.fetchone()[0]
                    profileShowPage(nextid,backhome=True)

          

    editCaseEditButton = tk.Button(editCaseFormFrame, text='ویرایش',font=('IRANSans',12),bg='#3fe095', highlightbackground='#EF7A52',activebackground='#EF7A52',activeforeground='white' ,foreground='white',padx=10,borderwidth=0,command=lambda: editCaseEdit(userData['id'],caseNameVar.get(),casePlaceVar.get(),caseTextVar.get(),casePhoneVar.get()))
    editCaseEditButton.grid(row=5,column=4, ipady=5,ipadx=5, sticky='nsew',padx=5,pady=5)

    editCaseDeleteButton = tk.Button(editCaseFormFrame, text='حذف',font=('IRANSans',12),bg='#F72929', highlightbackground='#EF7A52',activebackground='#EF7A52',activeforeground='white' ,foreground='white',padx=10,borderwidth=0,command=lambda: editCaseDelete(userData['id'],backhome))
    editCaseDeleteButton.grid(row=6,column=4, ipady=5,ipadx=5, sticky='nsew',padx=5,pady=5)

    editCaseBackForwardBtnFrame = tk.Frame(editCaseFormFrame,pady=5)
    editCaseBackForwardBtnFrame.config(background='white')
    editCaseBackForwardBtnFrame.columnconfigure(0,weight=1)
    editCaseBackForwardBtnFrame.columnconfigure(1,weight=1)
    cur = con.cursor()
    cur.execute("SELECT count(*) FROM user WHERE id >"+str(userData['id']))
    if cur.fetchone()[0]!=0:
        cur.execute("SELECT min(id) FROM user WHERE id >"+str(userData['id']))
        nextid = cur.fetchone()[0]
        editCaseForwardBtn = tk.Button(editCaseBackForwardBtnFrame, text='بعدی',font=('IRANSans',12),bg='#EF7A52', highlightbackground='#EF7A52',activebackground='#EF7A52',activeforeground='white' ,foreground='white',padx=10,borderwidth=0,command=lambda: [profileShowPage(nextid,backhome),profileShowWindow.destroy()])
        editCaseForwardBtn.grid(row=0,column=0, ipady=5,ipadx=5, sticky='nsew',padx=(0,5),pady=0)

    cur = con.cursor()
    cur.execute("SELECT count(*) FROM user WHERE id <"+str(userData['id']))
    if cur.fetchone()[0]!=0:
        cur.execute("SELECT max(id) FROM user WHERE id <"+str(userData['id']))
        lastid = cur.fetchone()[0]
        editCaseBackBtn = tk.Button(editCaseBackForwardBtnFrame, text='قبلی',font=('IRANSans',12),bg='#EF7A52', highlightbackground='#EF7A52',activebackground='#EF7A52',activeforeground='white' ,foreground='white',padx=10,borderwidth=0,command=lambda: [profileShowPage(lastid,backhome),profileShowWindow.destroy()])
        editCaseBackBtn.grid(row=0,column=1, ipady=5,ipadx=5, sticky='nsew',padx=(5,0),pady=0)
    
    editCaseBackForwardBtnFrame.grid(row=7,column=4, ipady=5,ipadx=5, sticky='nsew',padx=5,pady=0)
    if userData['frontphoto']!='':
        global frontPhotoimg
        frontimg = Image.open(os.getcwd()+f"/analyzed/{userData['id']}_front_result.jpg")
        frontimg.thumbnail((300,300))
        frontPhotoimg = ImageTk.PhotoImage(frontimg)
        frontImgPanel = tk.Label(profileShowWindow, image = frontPhotoimg)
        frontImgPanel.bind("<Button-1>", lambda e:zoomImage(os.getcwd()+f"/analyzed/{userData['id']}_front_result.jpg"))
        frontImgPanel.place(x=10, y=60)
    
    if userData['sidephoto']!='':
        global sidePhotoimg
        sideimg = Image.open(os.getcwd()+f"/analyzed/{userData['id']}_side_result.jpg")
        sideimg.thumbnail((300,300))
        sidePhotoimg = ImageTk.PhotoImage(sideimg)
        sideImgPanel = tk.Label(profileShowWindow, image = sidePhotoimg)
        sideImgPanel.bind("<Button-1>", lambda e:zoomImage(os.getcwd()+f"/analyzed/{userData['id']}_side_result.jpg"))
        sideImgPanel.place(x=320, y=60)


    def exportReport(name,place,address,text,frontImage,sideImage):
        
        my_screenshot = screenshot()
        my_screenshot.save(r"im.png")
        with Image.open('im.png') as img:
            # Define the cropping box (left, upper, right, lower)
            # Adjust the box according to your needs
            resized_img = img.resize((profileShowWindow.winfo_screenwidth()*2, profileShowWindow.winfo_screenheight()*2))
            left = 2*int(profileShowWindow.winfo_screenwidth()/2-500)
            upper = 2*int(profileShowWindow.winfo_screenheight()/2-300)
            right = 2*int(profileShowWindow.winfo_screenwidth()/2+500)
            lower = 2*int(profileShowWindow.winfo_screenheight()/2+350)
            cropped_img = resized_img.crop((left, upper, right, lower))

            draw = ImageDraw.Draw(cropped_img)
            draw.rectangle((1340, 360, 2000, 735), fill=(255, 255, 255))

            file = fd.asksaveasfile(mode='wb', defaultextension=".png")
            if file:
                cropped_img.save(file) # saves the image to the input file name.
            # Save the cropped image, overwriting the original file

    exportButton = tk.Button(titleBar,text='ذخیره فایل گزارش',font=('IRANSans',10),command=lambda:exportReport(caseNameVar.get(),casePlaceVar.get(),casePhoneVar.get(),caseTextVar.get(),userData['frontphoto'],userData['sidephoto']),highlightbackground='#EF7A52',activebackground='#EF7A52', foreground='#EF7A52',padx=10,borderwidth=0)
    exportButton.pack(side='left',padx=5)
    
    editCaseFormFrame.place(width=1000,y=50)

def addCasePage():
    uploadDir = tk.StringVar()
    uploadDir.set('/')
    addCaseWindow = tk.Toplevel()

    addCaseWindow.config(background='white')
    addCaseWindow.geometry('1000x700+'+str(int(addCaseWindow.winfo_screenwidth()/2-500))+'+'+str(int(addCaseWindow.winfo_screenheight()/2-350)))
    addCaseWindow.overrideredirect(True)


    titleBar = tk.Frame(addCaseWindow,background='#EF7A52')

    titleBar.place(relx=0,rely=0,height=50,relwidth=1)
    appTitle = tk.Label(titleBar,text='نرم افزار تشخیص ناهنجاری اسکلتی', font=('IRANSans Bold', 12),background='#EF7A52',foreground='white')
    appTitle.pack(side='right',padx=5)

    appExitButton = tk.Button(titleBar,text='صفحه اصلی',font=('IRANSans',10),command=lambda:addCaseWindow.destroy(),highlightbackground='#EF7A52',activebackground='#EF7A52', foreground='#EF7A52',padx=10,borderwidth=0)
    appExitButton.pack(side='left',padx=5)

    addCaseFormFrame = tk.Frame(addCaseWindow,pady=5,padx=5)
    addCaseFormFrame.config(background='white')
    addCaseFormFrame.columnconfigure(0,weight=1)
    addCaseFormFrame.columnconfigure(1,weight=1)
    addCaseFormFrame.columnconfigure(2,weight=1)
    addCaseFormFrame.columnconfigure(3,weight=1)
    addCaseFormFrame.columnconfigure(4,weight=1)
    caseNameVar = tk.StringVar()
    caseNameVar.set('نام فرد')
    addCaseNameEntry = tk.Entry(addCaseFormFrame,textvariable=caseNameVar,justify='right',bg='#e6e6e6',font=('IRANSans',10),borderwidth=0)    
    addCaseNameEntry.grid(row=0, column=4, ipady=5,ipadx=5,sticky='nsew',padx=5,pady=5)

    casePlaceVar = tk.StringVar()
    casePlaceVar.set('نام مرکز')
    addCasePlaceEntry = tk.Entry(addCaseFormFrame,textvariable=casePlaceVar,justify='right',bg='#e6e6e6',font=('IRANSans',10),borderwidth=0)    
    addCasePlaceEntry.grid(row=1, column=4, ipady=5,ipadx=5, sticky='nsew',padx=5,pady=5)

    caseTextVar = tk.StringVar()
    caseTextVar.set('توضیحات')
    addCasePlaceEntry = tk.Entry(addCaseFormFrame,textvariable=caseTextVar,justify='right',bg='#e6e6e6',font=('IRANSans',10),borderwidth=0)    
    addCasePlaceEntry.grid(row=2, column=4, ipady=5,ipadx=5, sticky='nsew',padx=5,pady=5)

    casePhoneVar = tk.StringVar()
    casePhoneVar.set('شماره تلفن')
    addCasePhoneEntry = tk.Entry(addCaseFormFrame,textvariable=casePhoneVar,justify='right',bg='#e6e6e6',font=('IRANSans',10),borderwidth=0)    
    addCasePhoneEntry.grid(row=3, column=4, ipady=5,ipadx=5, sticky='nsew',padx=5,pady=5)

    sidePhotoAdressVar = tk.StringVar()
    def sidePhoteBrowseFile():
        filetypes = (
            ('Image file', '*.jpg *.png *.jpeg'),
            ('All files', '*.*')
        )

        filename = fd.askopenfilename(
            title='Open a file',
            initialdir=uploadDir.get(),
            filetypes=filetypes,parent=addCaseWindow)
        uploadDir.set(os.path.dirname(filename))
        sidePhotoAdressVar.set(filename)
        addCaseSidePhoteButton.configure(bg='#86F78A',text='تصویر انتخاب شد',foreground='white')


        
    addCaseSidePhoteButton = tk.Button(addCaseFormFrame, text='تصویر از بغل',font=('IRANSans',10),bg='#e6e6e6', highlightbackground='#e6e6e6',activebackground='#e6e6e6',activeforeground='white' ,foreground='#EF7A52',padx=10,borderwidth=0,command=sidePhoteBrowseFile)
    addCaseSidePhoteButton.grid(row=5,column=4, ipady=5, sticky='nsew',padx=5,pady=5)

    frontPhotoAdressVar = tk.StringVar()
    def frontPhoteBrowseFile():
        filetypes = (
            ('Image file', '*.jpg *.png *.jpeg'),
            ('All files', '*.*')
        )

        filename = fd.askopenfilename(
            title='Open a file',
            initialdir=uploadDir.get(),
            filetypes=filetypes,parent=addCaseWindow)
        uploadDir.set(os.path.dirname(filename))
        frontPhotoAdressVar.set(filename)
        addCaseFrontPhoteButton.configure(bg='#86F78A',text='تصویر انتخاب شد',foreground='white')


        
    addCaseFrontPhoteButton = tk.Button(addCaseFormFrame, text='تصویر از روبرو',font=('IRANSans',10),bg='#e6e6e6', highlightbackground='#e6e6e6',activebackground='#e6e6e6',activeforeground='white' ,foreground='#EF7A52',padx=10,borderwidth=0,command=frontPhoteBrowseFile)
    addCaseFrontPhoteButton.grid(row=6,column=4, ipady=5, sticky='nsew',padx=5,pady=5)


    def addCaseInsert():
        current_id()
        cur = con.cursor()
        shutil.copy(frontPhotoAdressVar.get(), os.getcwd()+"/photos/"+str(current_id())+"_front.jpg")
        frontPhotoAdressVar.set(os.getcwd()+"/photos/"+str(current_id())+"_front.jpg")
        shutil.copy(sidePhotoAdressVar.get(), os.getcwd()+"/photos/"+str(current_id())+"_side.jpg")
        sidePhotoAdressVar.set(os.getcwd()+"/photos/"+str(current_id())+"_side.jpg")
        analyze_output = analyze_images(frontPhotoAdressVar.get(),sidePhotoAdressVar.get(),os.getcwd()+'/analyzed/')
        insert_values = [str(current_id()), caseNameVar.get() if caseNameVar.get()!= "نام فرد" else "", casePlaceVar.get() if casePlaceVar.get()!= "نام مرکز" else "",caseTextVar.get() if caseTextVar.get()!= "توضیحات" else "",casePhoneVar.get() if casePhoneVar.get()!= "شماره تلفن" else "",sidePhotoAdressVar.get(),frontPhotoAdressVar.get()]
        for key in analyze_output:
            insert_values.append(analyze_output[key])
        ky_lor_result = ky_lor(sidePhotoAdressVar.get())
        kyphosis = ky_lor_result['kyphosis']
        lordosis = ky_lor_result['lordosis']
        insert_values.append(lordosis)
        insert_values.append(kyphosis)
        cur.execute("""INSERT INTO user(id,name,place,description,phone,sidePhoto,frontPhoto,PelvicSlope, EyesSlope, ShoulderSlope, Parantezi, Zarbdari, RightKneeAngle, ForwardHeadSlope, PelvicSlopeAnalysis, EyesSlopeAnalysis, ShoulderSlopeAnalysis, ParanteziAnalysis, ZarbdariAnalysis, ForwardHeadSlopeAnalysis, Lordosis, Kyphosis) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", tuple(insert_values))
        con.commit() 
        updateTable()
        addCaseWindow.destroy()
        profileShowPage(cur.lastrowid)

    addCaseInsertButton = tk.Button(addCaseFormFrame, text='ثبت',font=('IRANSans',12),bg='#EF7A52', highlightbackground='#EF7A52',activebackground='#EF7A52',activeforeground='white' ,foreground='white',padx=10,borderwidth=0,command=addCaseInsert)
    addCaseInsertButton.grid(row=7,column=4, ipady=5,ipadx=5, sticky='nsew',padx=5,pady=5)

    logogimg2 = Image.open('logos.jpg')
    logogimg2.thumbnail((100,100))
    global logoPhotoimg2
    logoPhotoimg2 = ImageTk.PhotoImage(logogimg2)
    logoPhotoimgPanel2 = tk.Label(addCaseWindow, image = logoPhotoimg2,highlightthickness = 0,borderwidth=0,compound="center")
    logoPhotoimgPanel2.place(y=650,x=450)
    addCaseFormFrame.place(width=1000,y=50)

welcomeWindow = tk.Tk()
welcomeWindow.geometry('500x300+'+str(int(welcomeWindow.winfo_screenwidth()/2-250))+'+'+str(int(welcomeWindow.winfo_screenheight()/2-150)))
welcomeWindow.overrideredirect(True)
welcomeWindow.config(background='#EF7A52')
welcomeTitle = tk.Label(welcomeWindow,text='نرم افزار تشخیص ناهنجاری اسکلتی',font=('IRANSans Bold', 20),background='#EF7A52',foreground='white')
welcomeTitle.pack(pady=90)
welcomeDescription = tk.Label(welcomeWindow,text='این سیستم برای تشخیص ناهنجاری اسکلتی با استفاده از روش های هوشمند پردازش تصویر طراحی شده است',font=('IRANSans', 10), wraplength=350, justify="center",background='#EF7A52',foreground='white')
welcomeDescription.pack(pady=0)
welcomeWindow.after(1000, lambda: welcomeWindow.destroy()) 
welcomeWindow.mainloop()



window = tk.Tk()
window.config(background='white')
window.geometry('1000x700+'+str(int(window.winfo_screenwidth()/2-500))+'+'+str(int(window.winfo_screenheight()/2-350)))
window.overrideredirect(True)

titleBar(window,True)


addCasePageButton = tk.Button(window,text='افزودن فرد جدید',font=('IRANSans',12),bg='#EF7A52', highlightbackground='#EF7A52',activebackground='#EF7A52',activeforeground='white' ,foreground='white',padx=10,borderwidth=0,command=addCasePage)
addCasePageButton.place(x=845,y=60)
def callback(sv):
    updateTable(sv.get())
searchNameVar = tk.StringVar()
searchNameVar.set('جستجو')
searchNameVar.trace("w", lambda name, index, mode, sv=searchNameVar: callback(searchNameVar))
searchNameEntry = tk.Entry(window,textvariable=searchNameVar,justify='right',bg='#e6e6e6',font=('IRANSans',10),borderwidth=0,)    
searchNameEntry.place(x=635,y=60,height=43,width=200)

cur.execute('select * from user order by id desc limit 100')
dataset = cur.fetchall()

style = ttk.Style()
style.configure("Treeview.Heading", font=('IRANSans Bold',10))
style.configure("Treeview", font=('IRANSans',10))
style.configure('Treeview', borderwidth=0, relief="flat")

table = ttk.Treeview(window, columns = ('id','name', 'place', 'phone')[::-1], show = 'headings')
table.column('id',anchor='center')
table.column('name',anchor='center')
table.column('place',anchor='center')
table.column('phone',anchor='center')

table.heading('id', text = 'ردیف')
table.heading('name', text = 'نام فرد')
table.heading('place', text = 'نام مرکز')
table.heading('phone', text = 'شماره تلفن')
table.pack(fill = 'both', expand = True,pady=(110,10),padx=10)

# insert values into a table
# table.insert(parent = '', index = 0, values = ('John', 'Doe', 'JohnDoe@email.com'))
for i in dataset[::-1]:
	data = (i[0], i[1], i[2], i[4])[::-1]
	table.insert(parent = '', index = 0, values = data)


# events
def item_select(_):
    for i in table.selection():
        profileShowPage(table.item(i)['values'][-1],True)
	# table.item(table.selection())

# def delete_items(_):
#     for i in table.selection():
#         table.delete(i)
#         cur.execute("delete from user where id ="+str(table.item(i)['values'][-1]))
#         con.commit()


table.bind('<Double-1>', item_select)
#table.bind('<Delete>', delete_items)
logogimg = Image.open('logos.jpg')
logogimg.thumbnail((100,100))
logoPhotoimg = ImageTk.PhotoImage(logogimg)
logoPhotoimgPanel = tk.Label(window, image = logoPhotoimg,highlightthickness = 0,borderwidth=0,compound="center")
logoPhotoimgPanel.pack()

window.mainloop()
exit()
con.close()
import kaleido
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
import pyautogui

def file_save():
    f = fd.asksaveasfile(mode='w', defaultextension=".txt")
    if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    text2save = str(text.get(1.0, END)) # starts from `1.0`, not `0.0`
    f.write(text2save)
    f.close() # `()` was missing.
    


translateDict = {
    'PelvicSlope':'نابرابری لگن',
    'EyesSlope':'کج گردنی',
    'ShoulderSlope':'نابرابری شانه',
    'Parantezi':'زانوی پرانتزی',
    'Zarbdari':'زانوی ضربدری',
    'ForwardHeadSlope':'سر به جلو'
}
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
def analyze_images(front_image_path, side_image_path, output_folder):
    return({'PelvicSlope': 0.0502,
    'EyesSlope': -0.0171,
    'ShoulderSlope': 0.0535,
    'Parantezi': 0.741,
    'Zarbdari': 1.0203,
    'RightKneeAngle': 178.7878,
    'ForwardHeadSlope': -3.3143,
    'PelvicSlopeAnalysis': 'زیاد',
    'EyesSlopeAnalysis': 'نرمال',
    'ShoulderSlopeAnalysis': 'زیاد',
    'ParanteziAnalysis': 'زیاد',
    'ZarbdariAnalysis': 'نرمال',
    'ForwardHeadSlopeAnalysis': 'نرمال'})
def ky_lor(side_image_path):
    return({'lordosis': 3.5, 'kyphosis': 3.5})
con = sqlite3.connect(os.getcwd()+"/database8.db")
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
def titleBar(window,home=False,showProfile=False):
    titleBar = tk.Frame(window,background='#EF7A52')

    titleBar.place(relx=0,rely=0,height=50,relwidth=1)
    if showProfile:
        titleText = 'گزارش نرم افزار تشخیص ناهنجاری اسکلتی'
    else:
        titleText = 'نرم افزار تشخیص ناهنجاری اسکلتی'
    appTitle = tk.Label(titleBar,text=titleText, font=('IRANSans Bold', 12),background='#EF7A52',foreground='white')
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
    appTitle = tk.Label(titleBar,text='گزارش نرم افزار تشخیص ناهنجاری اسکلتی', font=('IRANSans Bold', 12),background='#EF7A52',foreground='white')
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
        guageimgs[i].thumbnail((250,250))
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
        guageImgPanels[i].place(x=i*230+10 if i<4 else (i-4)*230+10, y=370 if i<4 else 530)


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
        frontimg = Image.open(userData['frontphoto'])
        frontimg.thumbnail((300,300))
        frontPhotoimg = ImageTk.PhotoImage(frontimg)
        frontImgPanel = tk.Label(profileShowWindow, image = frontPhotoimg)
        frontImgPanel.bind("<Button-1>", lambda e:zoomImage(userData['frontphoto']))
        frontImgPanel.place(x=10, y=60)
    
    if userData['sidephoto']!='':
        global sidePhotoimg
        sideimg = Image.open(userData['sidephoto'])
        sideimg.thumbnail((300,300))
        sidePhotoimg = ImageTk.PhotoImage(sideimg)
        sideImgPanel = tk.Label(profileShowWindow, image = sidePhotoimg)
        sideImgPanel.bind("<Button-1>", lambda e:zoomImage(userData['sidephoto']))
        sideImgPanel.place(x=320, y=60)

    editCaseFormFrame.place(width=1000,y=50)
    def exportReport(name,place,address,text,frontImage,sideImage):
        # editCaseDeleteButton.destroy()
        # editCaseEditButton.destroy()
        # try:
        #     editCaseForwardBtn.destroy()
        # except:
        #     pass
        # try:
        #     editCaseBackBtn.destroy()
        # except:
        #     pass

        my_screenshot = pyautogui.screenshot()
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
            draw.rectangle((1360, 360, 2000, 635), fill=(255, 255, 255))


            file = fd.asksaveasfile(mode='wb', defaultextension=".pdf")
            if file:
                cropped_img.save(file) # saves the image to the input file name.
            # Save the cropped image, overwriting the original file
            
    exportButton = tk.Button(titleBar,text='ذخیره فایل گزارش',font=('IRANSans',10),command=lambda:[exportReport(caseNameVar.get(),casePlaceVar.get(),casePhoneVar.get(),caseTextVar.get(),userData['frontphoto'],userData['sidephoto'])],highlightbackground='#EF7A52',activebackground='#EF7A52', foreground='#EF7A52',padx=10,borderwidth=0)
    exportButton.pack(side='left',padx=5)
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
table.pack(fill = 'both', expand = True,pady=(110,0),padx=10)

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
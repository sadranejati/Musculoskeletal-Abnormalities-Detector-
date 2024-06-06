import tkinter as tk 
from tkinter import ttk 
from tkinter import filedialog as fd
import sqlite3
from PIL import ImageTk, Image

con = sqlite3.connect("databasee.db")
cur = con.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS user(
id INTEGER PRIMARY KEY,
name TEXT,
place Text,
description TEXT,
phone TEXT,
sidePhoto TEXT,
frontPhoto TEXT)
""")

def updateTable():   
    global table 
    global cur
    for row in table.get_children():
        table.delete(row)
    cur.execute('select * from user order by id desc limit 100')
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
    zoomImageWindow.title('بازنمایی تصویر')
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
    titleBar = tk.Frame(profileShowWindow,background='#EF7A52')

    titleBar.place(relx=0,rely=0,height=50,relwidth=1)
    appTitle = tk.Label(titleBar,text='نرم افزار تشخیص ناهنجاری اسکلتی', font=('IRANSans Bold', 12),background='#EF7A52',foreground='white')
    appTitle.pack(side='right',padx=5)
    if backhome:
        appExitButton = tk.Button(titleBar,text='صفحه اصلی',font=('IRANSans',10),command=lambda:[profileShowWindow.destroy()],highlightbackground='#EF7A52',activebackground='#EF7A52', foreground='#EF7A52',padx=10,borderwidth=0)
        appExitButton.pack(side='left',padx=5)
    else: 
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

    def editCaseEdit(id, name, place, text, phone):
        cur = con.cursor()
        cur.execute("UPDATE user SET name = ?, place = ?, description = ?, phone = ? WHERE id = ?",(name,place,text,phone,id))
        con.commit() 
        updateTable()

    def editCaseDelete(id):
        cur = con.cursor()
        cur.execute("delete from user where id ="+str(id))
        con.commit()
        updateTable()
        profileShowWindow.destroy()

    editCaseEditButton = tk.Button(editCaseFormFrame, text='ویرایش',font=('IRANSans',12),bg='#3fe095', highlightbackground='#EF7A52',activebackground='#EF7A52',activeforeground='white' ,foreground='white',padx=10,borderwidth=0,command=lambda: editCaseEdit(userData['id'],caseNameVar.get(),casePlaceVar.get(),caseTextVar.get(),casePhoneVar.get()))
    editCaseEditButton.grid(row=5,column=4, ipady=5,ipadx=5, sticky='nsew',padx=5,pady=5)

    editCaseDeleteButton = tk.Button(editCaseFormFrame, text='حذف',font=('IRANSans',12),bg='#F72929', highlightbackground='#EF7A52',activebackground='#EF7A52',activeforeground='white' ,foreground='white',padx=10,borderwidth=0,command=lambda: editCaseDelete(userData['id']))
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

def addCasePage():
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
            initialdir='/',
            filetypes=filetypes,parent=addCaseWindow)
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
            initialdir='/',
            filetypes=filetypes,parent=addCaseWindow)
        frontPhotoAdressVar.set(filename)
        addCaseFrontPhoteButton.configure(bg='#86F78A',text='تصویر انتخاب شد',foreground='white')


        
    addCaseFrontPhoteButton = tk.Button(addCaseFormFrame, text='تصویر از روبرو',font=('IRANSans',10),bg='#e6e6e6', highlightbackground='#e6e6e6',activebackground='#e6e6e6',activeforeground='white' ,foreground='#EF7A52',padx=10,borderwidth=0,command=frontPhoteBrowseFile)
    addCaseFrontPhoteButton.grid(row=6,column=4, ipady=5, sticky='nsew',padx=5,pady=5)


    def addCaseInsert():
        cur = con.cursor()
        cur.execute("""INSERT INTO user(name,place,description,phone,sidePhoto,frontPhoto) VALUES (?,?,?,?,?,?)""", (caseNameVar.get() if caseNameVar.get()!= "نام فرد" else "", casePlaceVar.get() if casePlaceVar.get()!= "نام مرکز" else "",caseTextVar.get() if caseTextVar.get()!= "توضیحات" else "",casePhoneVar.get() if casePhoneVar.get()!= "شماره تلفن" else "",sidePhotoAdressVar.get(),frontPhotoAdressVar.get()))
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


titleBar = tk.Frame(window,background='#EF7A52')

titleBar.place(relx=0,rely=0,height=50,relwidth=1)
appTitle = tk.Label(titleBar,text='نرم افزار تشخیص ناهنجاری اسکلتی', font=('IRANSans Bold', 12),background='#EF7A52',foreground='white')
appTitle.pack(side='right',padx=5)

appExitButton = tk.Button(titleBar,text='خروج',font=('IRANSans',10),command=lambda:window.quit(),highlightbackground='#EF7A52',activebackground='#EF7A52', foreground='#EF7A52',padx=10,borderwidth=0)
appExitButton.pack(side='left',padx=5)

addCasePageButton = tk.Button(window,text='افزودن فرد جدید',font=('IRANSans',12),bg='#EF7A52', highlightbackground='#EF7A52',activebackground='#EF7A52',activeforeground='white' ,foreground='white',padx=10,borderwidth=0,command=addCasePage)
addCasePageButton.place(x=845,y=60)

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

window.mainloop()
exit()
con.close()
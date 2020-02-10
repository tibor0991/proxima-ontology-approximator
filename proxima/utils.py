import tkinter
from tkinter import filedialog, messagebox

tkinter.Tk().withdraw()

filetypes = {
    'owl': ('OWL Ontology file', '.owl'),
    'csv': ('CSV Table', '.csv')
}


def open_file(title, type):
    _, extension = filetypes[type]
    path = tkinter.filedialog.askopenfilename(title=title, defaultextension=extension, filetypes=(filetypes[type],))
    return path


def save_file(title, type):
    _, extension = filetypes[type]
    path = tkinter.filedialog.asksaveasfilename(title=title, defaultextension=extension, filetypes=(filetypes[type],))
    return path


def ask_boolean(title, message):
    return messagebox.askyesno(title, message)

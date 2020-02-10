import tkinter
from tkinter import filedialog, messagebox, simpledialog

tkinter.Tk().withdraw()

filetypes = {
    'owl': ('OWL Ontology file', '.owl'),
    'csv': ('CSV Table', '.csv')
}


def open_file(title, type):
    _, extension = filetypes[type]
    path = filedialog.askopenfilename(title=title, defaultextension=extension, filetypes=(filetypes[type],))
    return path


def save_file(title, type):
    _, extension = filetypes[type]
    path = filedialog.asksaveasfilename(title=title, defaultextension=extension, filetypes=(filetypes[type],))
    return path


def ask_boolean(title, message):
    return messagebox.askyesno(title, message)


def ask_parameter(param_name, init):
    return simpledialog.askfloat(title="Insert parameter",
                                 prompt=param_name + '\n\rrange: [0, 1]:',
                                 minvalue=0.,
                                 maxvalue=1.,
                                 initialvalue=init)

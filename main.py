import tkinter as tk

from application import Application


def main():
    window_width, window_height = 1280, 720

    root = tk.Tk()
    root.title("Project AiPO")
    root.resizable(False, False)
    root.geometry("{}x{}".format(window_width, window_height))
    app = Application(master=root, window_width=window_width, window_height=window_height)
    app.mainloop()


if __name__ == '__main__':
    main()

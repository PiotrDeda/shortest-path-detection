import tkinter as tk
from tkinter import ttk
from PIL import Image
import numpy as np


def pillow_to_cv2(pillow_image):
    return np.array(pillow_image.convert('RGB'))[:, :, ::-1].copy()


def cv2_to_pillow(cv2_image):
    if len(cv2_image.shape) == 2:  # Obraz w skali szarości
        return Image.fromarray(cv2_image)
    else:  # Obraz RGB
        return Image.fromarray(cv2_image[:, :, ::-1])


# UI

def create_button_with_tooltip(self, frame, text, command, tooltip_text=None, disabled_var=None):
    button = tk.Button(frame, text="{}{}".format(text, " (?)".format(tooltip_text) if tooltip_text else ""),
                       command=command)
    button.config(bg="#323232", fg="white")
    button.pack(fill=tk.X)

    if disabled_var is not None:
        _handle_disabled_var_change(button, disabled_var)

    if tooltip_text:
        _handle_tooltip(self, button, tooltip_text)


def create_checkbox_with_tooltip(self, frame, text, var, callback, tooltip_text, disabled):
    checkbox = tk.Checkbutton(frame, text="{}{}".format(text, " (?)".format(tooltip_text) if tooltip_text else ""),
                              variable=var, state='disabled' if disabled else 'normal')
    checkbox.config(bg="#323232", fg="white")
    checkbox.pack(side="top", fill=tk.X)

    if callback:
        var.trace("w", callback)

    if tooltip_text:
        _handle_tooltip(self, checkbox, tooltip_text)


def create_label_with_toolip(self, frame, text, tooltip_text=None):
    label = tk.Label(frame, text="{}{}".format(text, " (?)".format(tooltip_text) if tooltip_text else ""), fg="white")
    label.config(bg="#323232", fg="white")
    label.pack(side="top", fill=tk.X)

    if tooltip_text:
        _handle_tooltip(self, label, tooltip_text)


def create_slider_with_tooltip(self, frame, text, var, from_, to, callback, tooltip_text):
    create_label_with_toolip(self, frame, text, tooltip_text)

    slider = tk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, variable=var)
    slider.config(bg="#323232", fg="white")
    slider.pack(side="top", fill=tk.X, padx=5, pady=5)

    if callback:
        var.trace("w", callback)


def create_select_with_tooltip(self, frame, text, var, options_var, callback=None, tooltip_text=None, disabled_var=None):
    create_label_with_toolip(self, frame, text, tooltip_text)

    combobox = ttk.Combobox(frame, textvariable=var)
    combobox.pack(side="top", fill=tk.X)

    if disabled_var is not None:
        _handle_disabled_var_change(combobox, disabled_var)

    if callback:
        var.trace("w", callback)

    # handle options

    def update_options(*args):
        combobox['values'] = options_var.get().split(",")

    # Update combobox values immediately
    update_options()

    # Update combobox values whenever options_var changes
    options_var.trace("w", update_options)


def get_image(self, image=None, image_path=None):
    img = image

    if image_path:
        img = Image.open(image_path)

    # Image aspect ratio
    max_width = self.window_width * 0.8
    max_height = self.window_height - self.title.winfo_height() - self.footer.winfo_height() - self.content_header.winfo_height() - 4
    width_ratio = max_width / img.width
    height_ratio = max_height / img.height
    min_ratio = min(width_ratio, height_ratio)

    # Dimensions based on aspect ratio
    new_width = int(img.width * min_ratio)
    new_height = int(img.height * min_ratio)

    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    return img


#

def _handle_tooltip(self, ui_element, tooltip_text):
    if tooltip_text:
        def enter(event):
            self.tooltip = tk.Toplevel(self)
            self.tooltip.wm_overrideredirect(True)  # Remove border
            label = tk.Label(self.tooltip, text=tooltip_text, fg="black", bg="white", relief="solid", borderwidth=1,
                             padx=4,
                             pady=4)
            label.pack()
            x = ui_element.winfo_rootx()
            y = ui_element.winfo_rooty()
            width = ui_element.winfo_width()
            self.tooltip.wm_geometry(f"+{x + width}+{y + 5}")

        def leave(event):
            self.tooltip.destroy()

        ui_element.bind("<Enter>", enter)
        ui_element.bind("<Leave>", leave)


def _handle_disabled_var_change(ui_element, var):
    def handle_change(*args):
        ui_element.config(state='disabled' if var.get() else 'normal')

    handle_change()
    var.trace("w", handle_change)

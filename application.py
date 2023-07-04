import platform
from tkinter import filedialog

from enum import Enum
from PIL import ImageTk

from dijkstra_algorithm import dijkstra
from graph import Graph
from procimg import ProcImg
from utils import *


class Application(tk.Frame):
    def __init__(self, master=None, window_width=1280, window_height=720):
        super().__init__(master)
        self.master = master
        self.config(bg="#606060")

        self.window_width = window_width
        self.window_height = window_height
        self.pack(fill=tk.BOTH, expand=True)

        self.text = None
        self.current_image_type = None
        self.image = None
        self.original_image = None
        self.transformed_image = None

        # Transformations
        self.binarization = tk.BooleanVar()
        self.binarization.set(True)
        self.segmentation = tk.BooleanVar()
        self.segmentation.set(True)
        self.morph = tk.BooleanVar()
        self.morph.set(True)
        self.skeletonization = tk.BooleanVar()
        self.skeletonization.set(True)
        self.branch_removal = tk.BooleanVar()
        self.branch_removal.set(True)

        # Graph
        self.graph = None
        self.graph_image = None
        self.font_size = tk.IntVar(value=20)
        self.path_start = tk.StringVar()
        self.path_end = tk.StringVar()
        self.path_options = tk.StringVar()
        self.path_options_trigger = tk.BooleanVar()
        self.path_options_trigger.set(True)

        # Header
        self.title = tk.Label(self, text="Find fastest path", bg='black', fg='white', padx=10, pady=10)
        self.title.pack(fill=tk.X)

        # Interface frame
        self.interface_frame = tk.Frame(self, width=self.window_width * 0.2, borderwidth=1, relief="solid")
        self.interface_frame.pack_propagate(False)
        self.interface_frame.pack(side="left", fill=tk.Y)
        self.should_disable_show_buttons = tk.BooleanVar()
        self.should_disable_show_buttons.set(True)
        self.should_disable_show_graph_path_points_select = tk.BooleanVar()
        self.should_disable_show_graph_path_points_select.set(True)
        self.should_disable_show_graph_with_shortest_path = tk.BooleanVar()
        self.should_disable_show_graph_with_shortest_path.set(True)
        self.create_buttons()
        self.interface_frame.config(bg="#606060")


        # Content header
        self.content_header = tk.Label(self, bg='grey', relief='solid', borderwidth=1, padx=8, pady=8, anchor='w',
                                       fg="black")
        self.content_header.pack(fill=tk.X)
        self.set_header_text("Load image first")

        # Content frame
        self.content_frame = tk.Frame(self, borderwidth=1, relief="solid")
        self.content_frame.config(bg="#606060")
        self.content_frame.pack(side="left", fill=tk.BOTH, expand=True)

        # Footer
        self.footer = tk.Label(self.master, text="Authors: Adam Łaba, Aleksander Kluczka, Kamil Jagodziński, Jakub Kraśniak, Piotr Deda, Krystian Śledź, Mirosław Kołodziej, Paweł Sipko", bg='black', fg='white', padx=10,
                               pady=10)
        self.footer.pack(side="bottom", fill=tk.X)

    # Methods

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if filepath:
            self.original_image = self.get_image(image_path=filepath)
            self.show_original_image()
            self.handle_transformation_change()
            self.graph = Graph()
            self.handle_graph_change()
            self.should_disable_show_buttons.set(False)

    def show_original_image(self):
        self.current_image_type = ImageType.ORIGINAL
        self.set_image(self.original_image)
        self.show_image()
        self.set_header_text("Original image")

    def show_transformed_image(self):
        self.current_image_type = ImageType.TRANSFORMED
        self.set_image(self.transformed_image)
        self.show_image()
        self.set_header_text("Transformed image")

    def handle_transformation_change(self, *args):
        img = ProcImg(pillow_to_cv2(self.original_image))
        if self.segmentation.get():
            img.segmentation()
        if self.binarization.get():
            img.binarization()
        if self.morph.get():
            img.morph_close()
        if self.skeletonization.get():
            img.skeletonization()
        if self.branch_removal.get():
            img.branch_removal()
        self.transformed_image = cv2_to_pillow(img.get_last_image())

        # self.graph = img.get_graph()
        # self.handle_graph_change()

        if self.current_image_type is ImageType.TRANSFORMED:
            self.show_transformed_image()

    def handle_font_size_change(self, *args):
        if self.current_image_type is ImageType.WITH_GRAPH:
            self.show_image_with_graph()

    def handle_graph_change(self):
        self.path_options.set(','.join([str(i) for i in range(len(self.graph.vertices))]))
        self.should_disable_show_graph_path_points_select.set(False)
        if self.current_image_type is ImageType.WITH_GRAPH:
            self.show_image_with_graph()

    def handle_path_point_change(self, *args):
        if self.path_start.get() and self.path_end.get():
            shortest_distance, shortest_path = dijkstra(self.graph, int(self.path_start.get()), int(self.path_end.get()))
            self.graph.set_shortest_path(shortest_path)
            self.should_disable_show_graph_with_shortest_path.set(False)
            if platform.system() != "Darwin" and self.current_image_type is ImageType.WITH_GRAPH_SP:
                self.show_image_with_graph_and_shortest_path()

    def show_image_with_graph(self):
        self.current_image_type = ImageType.WITH_GRAPH
        self.graph_image = self.graph.get_image_with_graph(self.original_image.copy(), self.font_size.get())
        self.set_image(self.graph_image)
        self.show_image()
        self.set_header_text("Original Image With Graph")

    def show_image_with_graph_and_shortest_path(self):
        self.current_image_type = ImageType.WITH_GRAPH_SP
        self.graph_image = self.graph.get_image_with_graph(self.original_image.copy(), self.font_size.get(), with_shortest_path=True)
        self.set_image(self.graph_image)
        self.show_image()
        self.set_header_text("Original Image With Graph And Shortest Path")

    # UI

    def create_buttons(self):
        self.create_label_with_toolip(self.interface_frame, text="1. Step - data")
        self.create_button_with_tooltip(self.interface_frame, text="Load Image", command=self.load_image)
        self.create_button_with_tooltip(self.interface_frame, text="Show Original Image",
                                        command=self.show_original_image, disabled_var=self.should_disable_show_buttons)
        self.create_divider(self.interface_frame)

        self.create_label_with_toolip(self.interface_frame, text="2. Step - transformations")
        self.create_checkbox_with_tooltip(self.interface_frame, text="Segmentation", var=self.segmentation,
                                          callback=self.handle_transformation_change,
                                          tooltip_text="Performs image segmentation using k-means clustering.")
        self.create_checkbox_with_tooltip(self.interface_frame, text="Binarization", var=self.binarization,
                                          callback=self.handle_transformation_change,
                                          tooltip_text="Converts an input image to a binary image using Otsu's thresholding.",
                                          disabled=True)
        self.create_checkbox_with_tooltip(self.interface_frame, text="Morph", var=self.morph,
                                          callback=self.handle_transformation_change,
                                          tooltip_text="Performs morphological closing on a binary image.")
        self.create_checkbox_with_tooltip(self.interface_frame, text="Skeletonization", var=self.skeletonization,
                                          callback=self.handle_transformation_change,
                                          tooltip_text="Performs skeletonization on a binary image.")
        self.create_checkbox_with_tooltip(self.interface_frame, text="Branch Removal", var=self.branch_removal,
                                          callback=self.handle_transformation_change,
                                          tooltip_text="Removes small branches from a skeletonized image.")
        self.create_button_with_tooltip(self.interface_frame, text="Show Transformed Image",
                                        command=self.show_transformed_image, disabled_var=self.should_disable_show_buttons)
        self.create_divider(self.interface_frame)

        self.create_label_with_toolip(self.interface_frame, text="3. Step - graph")
        self.create_slider_with_tooltip(self.interface_frame, "Font Size", self.font_size, callback=self.handle_font_size_change)
        self.create_button_with_tooltip(self.interface_frame, text="Show Image With Graph", command=self.show_image_with_graph, disabled_var=self.should_disable_show_buttons)
        self.create_select_with_tooltip(self.interface_frame, text="Select Start Point", var=self.path_start, options_var=self.path_options, callback=self.handle_path_point_change, disabled_var=self.should_disable_show_graph_path_points_select)
        self.create_select_with_tooltip(self.interface_frame, text="Select End Point", var=self.path_end, options_var=self.path_options, callback=self.handle_path_point_change, disabled_var=self.should_disable_show_graph_path_points_select)
        self.create_button_with_tooltip(self.interface_frame, text="Show Image With Shortest Path", command=self.show_image_with_graph_and_shortest_path, disabled_var=self.should_disable_show_graph_with_shortest_path)

    def show_image(self):
        self.clear_content()

        if self.image is None:
            self.set_text("No image to show")
            self.show_text(True)
            return

        image_label = tk.Label(self.content_frame, image=self.image)
        image_label.pack()

    def show_text(self, error=False):
        self.clear_content()

        if self.text is None:
            self.set_text("No text to show")
            self.show_text(True)
            return

        text_label = tk.Label(self.content_frame, text=self.text, font=("Arial", 20), fg="red" if error else "white")
        text_label.pack(anchor='w')

    def set_header_text(self, text):
        self.content_header.config(text=text)

    # UI utils

    def clear_content(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def create_button_with_tooltip(self, frame, text, command, tooltip_text=None, disabled_var=None):
        create_button_with_tooltip(self, frame, text, command, tooltip_text, disabled_var)

    def create_checkbox_with_tooltip(self, frame, text, var, callback=None, tooltip_text=None, disabled=False):
        create_checkbox_with_tooltip(self, frame, text, var, callback=callback, tooltip_text=tooltip_text,
                                     disabled=disabled)

    def create_divider(self, frame, color='black'):
        divider = tk.Frame(frame, bg=color, height=1)
        divider.pack(fill=tk.X, pady=8)

    def create_slider_with_tooltip(self, frame, text, var, from_=5, to=30, callback=None, tooltip_text=None):
        create_slider_with_tooltip(self, frame, text, var, from_=from_, to=to, callback=callback,
                                   tooltip_text=tooltip_text)

    def create_select_with_tooltip(self, frame, text, var, options_var, callback=None, tooltip_text=None, disabled_var=None):
        create_select_with_tooltip(self, frame, text, var, options_var, callback=callback, tooltip_text=tooltip_text, disabled_var=disabled_var)

    def create_label_with_toolip(self, frame, text, tooltip_text=None):
        create_label_with_toolip(self, frame, text, tooltip_text=tooltip_text)

    def get_image(self, image=None, image_path=None):
        return get_image(self, image=image, image_path=image_path)

    def set_image(self, image):
        self.image = ImageTk.PhotoImage(image)

    def set_text(self, text):
        self.text = text


class ImageType(Enum):
    ORIGINAL = 1
    TRANSFORMED = 2
    WITH_GRAPH = 3
    WITH_GRAPH_SP = 4
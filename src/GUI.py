import os
import customtkinter as ctk
from tkinter import filedialog, ttk
import cv2 as cv
from PIL import ImageTk, Image
from ImageController import ImageController, ImageObj
import imageProcessing as ip
import FeaturePlot as fp

images = ImageController() 
target = None
file_paths = []
colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']

# function to enable or disable all widgets within a container
# new_state can equal 'normal' or 'disabled'
def change_children_state(parent, new_state):
    for child in parent.winfo_children():
        child.configure(state=new_state)

# resize the image to fit within the image label but keep original proportions
# i.e. rescale the image to fit within the label, returns new dimensions
def smart_dimensions(image_shape, label_width, label_height):
    
    height, width = image_shape[0], image_shape[1]
    
    width_scale = label_width/width
    height_scale = label_height/height
    
    selec = min(width_scale, height_scale)
    
    new_shape = (int(width*selec), int(height*selec))
    
    return new_shape

# Convert image from cv Mat (numpy array) to GUI compatible form
def convert_image(image):
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    return image


# A reusable frame to control the selection of multiple images
class ImageSelectorFrame(ctk.CTkFrame):
    
    # Class constructor - places all widgets on the GUI
    def __init__(self, container):
        super().__init__(container)
        
        self.container = container
        
        self.configure(border_color='green', border_width=5)
        
        # ============= configure grid (3x3) =============
        self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure(3, weight=1)
        
        # ============= Widgets within this frame =============
        self.slider_var = ctk.IntVar(value=0)
        self.image_slider = ctk.CTkSlider(master=self, variable=self.slider_var, command=self.slider_func)
        self.image_slider.grid(row=1, column=2)
        
        self.left_button = ctk.CTkButton(master=self, text='<', command=lambda: self.button_func(-1), width=50)
        self.left_button.grid(row=1, column=1)
        
        self.right_button = ctk.CTkButton(master=self, text='>', command=lambda: self.button_func(1), width=50)
        self.right_button.grid(row=1, column=3)
        
        self.option_menu_var = ctk.StringVar(value='')
        self.file_option_menu = ctk.CTkOptionMenu(master=self, variable=self.option_menu_var, command=self.optionmenu_func)
        self.file_option_menu.grid(row=2, column=2)
        
        self.file_number_label = ctk.CTkLabel(master=self, text='')
        self.file_number_label.grid(row=3, column=2)
        
        # disable all widgets in this frame by default
        change_children_state(parent=self, new_state='disabled')
    
    # Updates the index using the slider
    def slider_func(self, new_index):
        if self.index != new_index:
            self.index = int(new_index)
            self.update(source='slider')

    # Updates the index using left or right arrow buttons
    def button_func(self, value):
        if (self.index + value) >= 0 and (self.index + value) < len(self.paths):
            self.index = self.index + value
            self.update(source='button')
    
    # Updates the index using the drop-down option menu of file names
    def optionmenu_func(self, selection):
        self.index = self.file_names.index(selection)
        self.update(source='optionmenu')
    
    
    # Takes argument 'source' - the widget that initiates the update. All other widgets will be updated to match the new selection
    def update(self, source):
        
        if source == 'optionmenu' or source == 'button':
            self.slider_var.set(self.index)
        
        if source == 'slider' or source == 'button':
            self.option_menu_var.set(self.file_names[int(self.index)])
        
        self.file_number_label.configure(text=f'Image {self.index+1} of {len(self.paths)}')
        self.container.update_image(self.index)

    
    # Function to activate the widgets within this frame and initialize variables
    def activate(self, paths):
        self.paths = paths
        self.file_names = [os.path.basename(path) for path in paths]
        self.index = 0
        
        if len(paths) > 1:
            self.image_slider.configure(from_=0, to=len(paths)-1, number_of_steps=len(paths)-1)
        
        self.file_option_menu.configure(values=self.file_names)
        
        self.slider_var.set(self.index)
        self.option_menu_var.set(self.file_names[int(self.index)])
        self.file_number_label.configure(text=f'Image {self.index+1} of {len(self.paths)}')
        
        change_children_state(parent=self, new_state='normal')
    
    # Function to deactivate widgets within this frame - used for when there is only one image selected
    def deactivate(self):
        self.index = 0        
        change_children_state(parent=self, new_state='disabled')


# A frame to control the selection of different steps of image processing (used within BatchSaveFrame)
class OptionFrame(ctk.CTkFrame):
    
    # Class constructor - places all widgets on the GUI
    def __init__(self, container):
        super().__init__(container)
        
        self.container = container
        
        # ============= Widgets within this frame =============
        self.options_label = ctk.CTkLabel(master=self, text='Options', bg_color='green')
        self.options_label.grid(row=1, column=1, padx=10, pady=10)
        
        self.original_check_box = ctk.CTkCheckBox(master=self, text='Original', state='disabled')
        self.original_check_box.select()
        self.original_check_box.grid(row=2, column=1, padx=10, pady=10, sticky='W')
        
        check_box_options = ['Cropped', 'Normalized', 'Thresholded', 'Annotated', 'Verticle Line', 'Rotated', 'Final']
        self.check_boxes = []
        self.check_var = ctk.StringVar()
        for i, option in enumerate(check_box_options):
            self.check_boxes.append(ctk.CTkCheckBox(master=self, text=option, variable=self.check_var, onvalue=str(i)+'1', offvalue=str(i)+'0', command=None, state='disabled'))
            self.check_boxes[-1].select()
            self.check_boxes[-1].grid(row=i+3, column=1, padx=10, pady=10, sticky='W')
        
        self.up_btn = ctk.CTkButton(master=self, text='↑', width=30, command=lambda : self.btn_func(-1), state='disabled')
        self.up_btn.grid(row=2, column=2, padx=5, pady=5)
        
        self.down_btn = ctk.CTkButton(master=self, text='↓', width=30, command=lambda : self.btn_func(1), state='disabled')
        self.down_btn.grid(row=9, column=2, padx=5, pady=5)
    
    
    # Controls the cumulative selection/deselection of the option checkboxes
    def check_box_func(self):
        
        # disable the command function to prevent recursive calls when using set()
        for check_box in self.check_boxes:
            check_box.configure(command=None)
        
        get = self.check_var.get()
        selection, selected = int(get[0]), bool(int(get[1]))
        
        if selected:
            for check_box in reversed(self.check_boxes[:selection]):
                self.check_var.set(check_box.onvalue)
        else:
            for check_box in self.check_boxes[selection:]:
                self.check_var.set(check_box.offvalue)
        
        # re-anable command function
        for check_box in self.check_boxes:
            check_box.configure(command=self.check_box_func)
        
        self.container.refresh()
    
    # Controls the step-wise selection/deselection of checkboxes using the up and down buttons
    def btn_func(self, change):
        
        if change == -1: 
            #deselect the last selected item
            for check_box in reversed(self.check_boxes):
                if check_box.get() == check_box.onvalue:
                    check_box.deselect()
                    break
            
        elif change == 1: 
            #select the first unselected item
            for check_box in self.check_boxes:
                if check_box.get() == check_box.offvalue:
                    check_box.select()
                    break
    
    
    # Function to activate the widgets within this frame
    def activate(self):
        
        for check_box in self.check_boxes:
            check_box.configure(state='normal', command=self.check_box_func)
        
        self.up_btn.configure(state='normal')
        self.down_btn.configure(state='normal')
    
    # Returns the selection state of the object
    def get_selection_state(self):
        
        for i, check_box in reversed(list(enumerate(self.check_boxes))):
            if int(check_box.get()[1]):
                return i
        else:
            return -1


# A frame to display images clustered by some attribute
class ClusterFrame(ctk.CTkFrame):
    
    # Class constructor - places all widgets on the GUI
    def __init__(self, container):
        super().__init__(container)
        
        self.container = container
        
        self.IMAGE_LABEL_WIDTH = 600
        self.IMAGE_LABEL_HEIGHT = 600
        
        self.configure(border_color='red')
        
        # ============= Widgets within this frame =============
        self.plot_name_label = ctk.CTkLabel(master=self, text='Plot of:\n', bg_color='red')
        self.plot_name_label.grid(row=0, column=1)
        self.plot_label = ctk.CTkLabel(master=self, text='plot goes here', bg_color='red', width=self.IMAGE_LABEL_WIDTH, height=self.IMAGE_LABEL_HEIGHT)
        self.plot_label.grid(row=1, column=1, padx=10, pady=10)
        
        self.image_name_label = ctk.CTkLabel(master=self, text='Image Name:\n', bg_color='red')
        self.image_name_label.grid(row=0, column=4, padx=10, pady=10)
        self.image_label = ctk.CTkLabel(master=self, text='image goes here', bg_color='blue', width=self.IMAGE_LABEL_WIDTH, height=self.IMAGE_LABEL_HEIGHT)
        self.image_label.grid(row=1, column=3, columnspan=3, padx=10, pady=10)
        
        values = ['Splay coefficient', 'Hand size', 'Contour alignment error (sum)', 'Contour alignment error (mean)', 'Contour alignment error (standev)', 
                    'Fingertip alignment error (sum)', 'Fingertip alignment error (mean)', 'Fingertip alignment error (standev)']
        self.feature_menu_var = ctk.StringVar(value=values[0])
        self.feature_menu = ctk.CTkOptionMenu(master=self, values=values, variable=self.feature_menu_var, command=self.cluster_func)
        self.feature_menu.grid(row=2, column=2, padx=10, pady=10)
        
        self.cluster_btn = ctk.CTkButton(master=self, text='Cluster data', command=self.cluster_func)
        self.cluster_btn.grid(row=3, column=2, padx=10, pady=10)
        
        self.plus_index_btn = ctk.CTkButton(master=self, text='>', state='disabled', command=lambda: self.btn_func(1))
        self.plus_index_btn.grid(row=2, column=5)
        self.minus_index_btn = ctk.CTkButton(master=self, text='<', state='disabled', command=lambda: self.btn_func(-1))
        self.minus_index_btn.grid(row=2, column=3)
        
        self.cluster_number_label = ctk.CTkLabel(master=self, text='')
        self.cluster_number_label.grid(row=2, column=4, padx=10, pady=10)
        
        self.cluster_number_var = ctk.IntVar()
        self.cluster_number_option = ctk.CTkOptionMenu(master=self, variable=self.cluster_number_var, state='disabled', command=self.select_cluster_func)
        self.cluster_number_option.grid(row=3, column=4, padx=10, pady=10)
        
        self.save_cluster_btn = ctk.CTkButton(master=self, text='Save Cluster', state='disabled', command=self.save_cluster)
        self.save_cluster_btn.grid(row=4, column=4, padx=10, pady=10)
        
        self.processing_label = ctk.CTkLabel(master=self, text='Select hand feature:')
        self.processing_label.grid(row=2, column=1, sticky='e')
    
    
    # Update the plot and images displayed on the screen
    def update(self):
        
        if hasattr(self, 'current_plot_array'):
            self.current_plot_array = cv.resize(self.current_plot_array, smart_dimensions(self.current_plot_array.shape, self.IMAGE_LABEL_WIDTH, self.IMAGE_LABEL_HEIGHT))
            self.current_plot_image = convert_image(self.current_plot_array)
            self.plot_label.configure(image=self.current_plot_image)
            
            self.plot_name_label.configure(text='Plot of:\n' + self.feature_menu_var.get())
        
        
        if hasattr(self, 'current_cluster'):
            self.image_name_label.configure(bg_color=colors[self.cluster_number_var.get()-1], 
            text='Image Name:' + self.current_cluster[self.index].get_file_name() + '\nin ' + colors[self.cluster_number_var.get()-1] + ' cluster')
            
            self.current_image = self.current_cluster[self.index].get_final_processed_image()
            self.current_image = cv.resize(self.current_image, smart_dimensions(self.current_image.shape, self.IMAGE_LABEL_WIDTH, self.IMAGE_LABEL_HEIGHT))
            self.current_image = convert_image(self.current_image)
            self.image_label.configure(image=self.current_image)
            self.image_label.configure(bg_color=colors[self.cluster_number_var.get()-1])
            
            self.cluster_number_label.configure(text='Displaying Image ' + str(self.index+1) + ' of ' + str(len(self.current_cluster))
            + ' in Cluster ' + str(self.cluster_number_var.get()) + ' of ' + str(self.number_of_clusters))
    
    
    # Function to select number of clusters, cluster the data and display plot
    def cluster_func(self, *_):
        self.processing_label.configure(text='Processing...')
        # Select the number of clusters the data will be split into
        try:
            while True:
                self.number_of_clusters = int(ctk.CTkInputDialog(master=self, text='Number of clusters (max 7)', title='Number of clusters').get_input())
                if self.number_of_clusters <= images.length() and self.number_of_clusters < 8 and self.number_of_clusters > 0:
                    break
        except TypeError:
            self.processing_label.configure(text='Select hand feature:')
            return
        except ValueError:
            self.processing_label.configure(text='Select hand feature:')
            return
        
        # Compute K-Means clusters and generate plot
        def plot_data(method, optional_param1=None, optional_param2=None):
            
            # Creates dictionary of ImageObj : image attribute 
            dic = {}
            for i in range(images.length()):
                imgobj = images.get_imageObj(i)
                if not imgobj.get_rejected():
                    dic[imgobj] = getattr(imgobj, method)(optional_param1, optional_param2)
            
            self.plotter = fp.FeaturePlot(dic, self.number_of_clusters)
            
            self.clustered_objects = self.plotter.get_clustered_objects()
            self.clustered_objects.sort(key=lambda x: getattr(x[0], method)(optional_param1, optional_param2))
            
            self.current_plot_array = self.plotter.get_plot()
            
        
        if self.feature_menu_var.get() == 'Hand size':
            plot_data('get_hand_size')
            self.feature_func = 'get_hand_size'
        elif self.feature_menu_var.get() == 'Splay coefficient':
            plot_data('get_splay_coefficient')
        elif self.feature_menu_var.get() == 'Contour alignment error (sum)':
            plot_data('get_contour_alignment_error', target, 'sum')
        elif self.feature_menu_var.get() == 'Contour alignment error (mean)':
            plot_data('get_contour_alignment_error', target, 'mean')
        elif self.feature_menu_var.get() == 'Contour alignment error (standev)':
            plot_data('get_contour_alignment_error', target, 'standev')
        elif self.feature_menu_var.get() == 'Fingertip alignment error (sum)':
            plot_data('get_fingertip_alignment_error', target, 'sum')
        elif self.feature_menu_var.get() == 'Fingertip alignment error (mean)':
            plot_data('get_fingertip_alignment_error', target, 'mean')
        elif self.feature_menu_var.get() == 'Fingertip alignment error (standev)':
            plot_data('get_fingertip_alignment_error', target, 'standev')
        
        
        # Activate relevant widgets
        self.cluster_number_option.configure(state='normal', values=[str(num) for num in range(1, self.number_of_clusters+1)])
        self.cluster_number_var.set(1)
        self.select_cluster_func(self.cluster_number_var.get())
        self.plus_index_btn.configure(state='normal')
        self.minus_index_btn.configure(state='normal')
        self.save_cluster_btn.configure(state='normal')
        
        self.cluster_btn.configure(text='Recluster data')
        self.processing_label.configure(text='Select hand feature:')
        
        self.update()
    
    
    # Function for selecting and viewing specific cluster number
    def select_cluster_func(self, selection):
        cluster_num = int(selection)-1
        self.current_cluster = self.clustered_objects[cluster_num]
        self.index = 0
        self.update()
    
    
    # Used for cycling through the cluster images using left and right arrow buttons on screen
    def btn_func(self, delta):
        
        self.index += delta
        if self.index > len(self.current_cluster)-1: 
            self.index = len(self.current_cluster)-1
        elif self.index < 0:
            self.index = 0
        
        self.update()
    
    
    # Saves images in selected cluster
    def save_cluster(self):
        save_path = filedialog.askdirectory()
        for image in self.current_cluster:
            image.save_image(save_path)
        print('save complete')


# A frame to display batch of images overlayed with target image
class OverlayFrame(ctk.CTkFrame):
    
    # Class constructor - places all widgets on the GUI
    def __init__(self, container):
        super().__init__(container)
        
        self.configure(border_color='red', border_width=5)
        
        self.container = container
        
        self.IMAGE_LABEL_WIDTH = 500
        self.IMAGE_LABEL_HEIGHT = 500
        self.index = 0
        
        # ============= configure grid (3x7) =============
        self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure(7, weight=1)
        
        # ============= Widgets within this frame =============
        self.processed_label = ctk.CTkLabel(master=self, text='Processed Image\n', bg_color='red')
        self.processed_label.grid(row=1, column=1, padx=10, pady=10)
        
        self.processed_image_label = ctk.CTkLabel(master=self, text='processed image goes here', bg_color='red', width=self.IMAGE_LABEL_WIDTH, height=self.IMAGE_LABEL_HEIGHT)
        self.processed_image_label.grid(row=2, column=1, rowspan=4, padx=10, pady=10, sticky='nw')
        
        self.overlay_label = ctk.CTkLabel(master=self, text='Overlayed Image\n', bg_color='purple')
        self.overlay_label.grid(row=1, column=2, padx=10, pady=10)
        
        self.overlay_image_label = ctk.CTkLabel(master=self, text='overlayed image goes here', bg_color='purple', width=self.IMAGE_LABEL_WIDTH, height=self.IMAGE_LABEL_HEIGHT)
        self.overlay_image_label.grid(row=2, column=2, rowspan=4, padx=10, pady=10, sticky='nw')
        
        self.target_label = ctk.CTkLabel(master=self, text='Target Image\n', bg_color='blue')
        self.target_label.grid(row=1, column=3, padx=10, pady=10)
        
        self.target_image_label = ctk.CTkLabel(master=self, text='target image goes here', bg_color='blue', width=self.IMAGE_LABEL_WIDTH, height=self.IMAGE_LABEL_HEIGHT)
        self.target_image_label.grid(row=2, column=3, rowspan=4, padx=10, pady=10, sticky='nw')
        
        self.image_selector_frame = ImageSelectorFrame(container=self)
        self.image_selector_frame.grid(row=6, column=2, sticky='n')
        
        self.select_target_btn = ctk.CTkButton(master=self, text='Select new target', command=self.select_target)
        self.select_target_btn.grid(row=6, column=3)
        
        self.align_error_label = ctk.CTkLabel(master=self, text='Contour alignment Error:')
        self.align_error_label.grid(row=7, column=2)
        
        while True:
            self.top = ctk.CTkToplevel(self)
            self.top.title('')
            self.top.geometry("200x200")
            self.top.protocol("WM_DELETE_WINDOW", lambda: None)
            self.select = ctk.CTkButton(master=self.top, text='Select a target image', command=self.select_target)
            self.select.pack(padx=20, pady=20)
            self.quit = ctk.CTkButton(master=self.top, text='Quit', command=lambda: os._exit(0))
            self.quit.pack(padx=20, pady=20)
            self.wait_window(self.top)
            if target is not None:
                break
    
    
    # Selects the desired target image and displays it in the right panel
    def select_target(self):
        self.top.destroy()
        file_path = filedialog.askopenfilename(filetypes=[('image files', '*.png')])
        if len(file_path) < 1:
            return
        
        global target
        target = ImageObj(file_path)        
        self.refresh()
        self.update_image(self.index)
    
    
    # Refeshes the GUI to update whats displayed
    def refresh(self):
        
        self.index = 0
        
        if images.length() > 0:
            self.image_selector_frame.activate(file_paths)
            if images.length() == 1:
                self.image_selector_frame.deactivate()
            self.update_image(self.index)
    
    
    # Updates the image displayed to the new selected image based on the given index
    def update_image(self, new_index):
        
        self.current_processed_image = images.get_final_processed_image(new_index)
        self.current_processed_image = cv.resize(self.current_processed_image, smart_dimensions(self.current_processed_image.shape, self.IMAGE_LABEL_WIDTH, self.IMAGE_LABEL_HEIGHT))
        
        if target is not None:
            self.current_target_image = target.get_final_processed_image()
            self.current_target_image = cv.resize(self.current_target_image, smart_dimensions(self.current_target_image.shape, self.IMAGE_LABEL_WIDTH, self.IMAGE_LABEL_HEIGHT))
            
            self.current_overlay_image = ip.overlay_images(self.current_processed_image, self.current_target_image) # aligned images must be same size
            
            self.current_target_image = convert_image(self.current_target_image)
            self.target_image_label.configure(image=self.current_target_image)
            self.target_label.configure(text='Target Image\n' + target.get_file_name())
            
            self.current_overlay_image = cv.resize(self.current_overlay_image, smart_dimensions(self.current_overlay_image.shape, self.IMAGE_LABEL_WIDTH, self.IMAGE_LABEL_HEIGHT))
            self.current_overlay_image = convert_image(self.current_overlay_image)
            self.overlay_image_label.configure(image=self.current_overlay_image)
            self.overlay_label.configure(text='Overlayed Image\n' + images.get_file_name(new_index) + ' + ' + target.get_file_name())
            
            alignment_error = images.get_imageObj(new_index).get_contour_alignment_error(target, 'sum')
            self.align_error_label.configure(text='Contour alignment Error: ' + str(alignment_error))
        
        self.current_processed_image = convert_image(self.current_processed_image)
        self.processed_image_label.configure(image=self.current_processed_image)
        self.processed_label.configure(text='Processed Image\n' + images.get_file_name(new_index))
        
        
        self.index = new_index


# A frame for selecting a batch of images, viewing the stages of image processing and saving them to disk
class BatchSaveFrame(ctk.CTkFrame):
    
    # Class contructor, places all widgets on the GUI
    def __init__(self, container):
        super().__init__(container)
        
        self.configure(border_color='red', border_width=5)
        
        self.container = container
        
        self.IMAGE_LABEL_WIDTH = 500
        self.IMAGE_LABEL_HEIGHT = 500
        
        # ============= configure grid (4x7) =============
        self.grid_columnconfigure(4, weight=1)
        self.grid_rowconfigure(7, weight=1)
        
        # ============= Widgets within this frame =============
        self.options_frame = OptionFrame(container=self)
        self.options_frame.grid(row=1, column=4, rowspan=4, padx=10, pady=10)
        
        self.original_label = ctk.CTkLabel(master=self, text='Original Image\n', bg_color='red')
        self.original_label.grid(row=1, column=1, padx=10, pady=10)
        
        self.original_image_label = ctk.CTkLabel(master=self, text='original image goes here', bg_color='red', width=self.IMAGE_LABEL_WIDTH, height=self.IMAGE_LABEL_HEIGHT)
        self.original_image_label.grid(row=2, column=1, rowspan=4, padx=10, pady=10, sticky='nw')
        
        self.processed_label = ctk.CTkLabel(master=self, text='Processed Image\n', bg_color='blue')
        self.processed_label.grid(row=1, column=3, padx=10, pady=10)
        
        self.processed_image_label = ctk.CTkLabel(master=self, text='processed image goes here', bg_color='blue', width=self.IMAGE_LABEL_WIDTH, height=self.IMAGE_LABEL_HEIGHT)
        self.processed_image_label.grid(row=2, column=3, rowspan=4, padx=10, pady=10, sticky='nw')
        
        self.select_files_btn = ctk.CTkButton(master=self, text='Select new batch', command=self.select_files)
        self.select_files_btn.grid(row=6, column=1, padx=10, pady=10)
        
        self.save_files_btn = ctk.CTkButton(master=self, text='Save files', state='disabled',command=self.save_files)
        self.save_files_btn.grid(row=6, column=3, padx=10, pady=10)
        
        self.image_selector_frame = ImageSelectorFrame(container=self)
        self.image_selector_frame.grid(row=6, column=2, padx=10, pady=10, sticky='n')
        
        self.reject_var = ctk.StringVar(value='')
        self.reject_switch = ctk.CTkSwitch(master=self, textvariable=self.reject_var, variable=self.reject_var, state='disabled', onvalue='Accepted', offvalue="Rejected", command=self.reject_switch_func)
        self.reject_switch.grid(row=5, column=4, padx=10, pady=10, sticky='n')
        
        while True:
            self.top = ctk.CTkToplevel(self)
            self.top.title('')
            self.top.geometry("200x200")
            self.top.protocol("WM_DELETE_WINDOW", lambda: None)
            self.select = ctk.CTkButton(master=self.top, text='Select a batch of files', command=self.select_files)
            self.select.pack(padx=20, pady=20)
            self.quit = ctk.CTkButton(master=self.top, text='Quit', command=lambda: os._exit(0))
            self.quit.pack(padx=20, pady=20)
            self.wait_window(self.top)
            if images.length() > 0:
                break
    
    
    # Opens file selection window for the user
    def select_files(self):
        self.top.destroy()
        global file_paths
        #Will only display .png image files when browsing for files
        file_paths = filedialog.askopenfilenames(filetypes=[('image files', '*.png')])
        
        if len(file_paths) < 1:
            return
        
        global images
        images = ImageController()
        
        images.load_images(file_paths)
        
        # Activate relevant widgets
        self.image_selector_frame.activate(file_paths)
        if images.length() == 1:
            self.image_selector_frame.deactivate()
        
        self.options_frame.activate()
        self.save_files_btn.configure(state='normal')
        self.set_switch()
        self.reject_switch.configure(state='normal')
        self.container.files_selected()
        self.update_image(0)
    
    
    # Prompts the user to choose a file path to save to 
    def save_files(self):
        save_path = filedialog.askdirectory()
        if len(save_path) < 1:
            return
        images.save_images(save_path)
    
    
    # Toggle Rejected/Accepted switch on and off without triggering the command function
    def set_switch(self, new_value='Accepted'):
        self.reject_switch.configure(command=None)
        self.reject_var.set(value=new_value)
        self.reject_switch.configure(command=self.reject_switch_func)
    
    
    # Refreshes frame to update whats displayed
    def refresh(self):
        if hasattr(self, 'index'):
            self.update_image(self.index)
    
    # Updates the image displayed on the frame based on the new index value provided
    def update_image(self, new_index):
        self.current_original_image = images.get_original_image(new_index)
        self.current_original_image = cv.resize(self.current_original_image, smart_dimensions(self.current_original_image.shape, self.IMAGE_LABEL_WIDTH, self.IMAGE_LABEL_HEIGHT))
        self.current_original_image = convert_image(self.current_original_image)
        self.original_image_label.configure(image=self.current_original_image)
        self.original_label.configure(text='Original Image\n' + images.get_file_name(new_index))
        
        self.current_processed_image = images.get_processed_image(new_index, self.options_frame.get_selection_state())
        self.current_processed_image = cv.resize(self.current_processed_image, smart_dimensions(self.current_processed_image.shape, self.IMAGE_LABEL_WIDTH, self.IMAGE_LABEL_HEIGHT))
        self.current_processed_image = convert_image(self.current_processed_image)
        self.processed_image_label.configure(image=self.current_processed_image)
        self.processed_label.configure(text='Processed Image\n' + images.get_file_name(new_index))
        
        # Toggles Accepted/Rejected switch
        if images.get_rejected(new_index):
            self.set_switch(self.reject_switch.offvalue)
        else:
            self.set_switch(self.reject_switch.onvalue)
        
        self.index = new_index
    
    # Sets the images to rejected by the user, image will not be saved
    def reject_switch_func(self):        
        images.toggle_rejected(self.index)


# Container to hold the 3 frames with a tabbed selection at the bottom of the screen
class NoteBook(ttk.Notebook):
    
    # Class constructor
    def __init__(self, container, width, height):
        super().__init__(container)
        
        customed_style = ttk.Style()
        customed_style.configure('Custom.TNotebook.Tab', padding=[12, 12], font=('Helvetica', 10))
        customed_style.configure('Custom.TNotebook', tabposition='sw')
        
        self.configure(width=width, height=height, style='Custom.TNotebook')
        
        # ============= major frames/use cases =============
        self.batch_save_frame = BatchSaveFrame(container=self)
        self.cluster_frame = ClusterFrame(container=self)
        self.overlay_frame = OverlayFrame(container=self)
        # self.batch_save_frame = BatchSaveFrame(container=self)
        
        # Pack the frames on the screen
        self.cluster_frame.pack(fill='both', expand=1)
        self.overlay_frame.pack(fill='both', expand=1)
        self.batch_save_frame.pack(fill='both', expand=1)
        
        # Add the frames as tabs
        self.add(self.batch_save_frame, text='Batch Processing')
        self.add(self.overlay_frame, text='Overlay and alignment')
        self.add(self.cluster_frame, text='Cluster and search')
    
    def files_selected(self):
        if hasattr(self, 'overlay_frame'):
            self.overlay_frame.refresh()


# Root window
class Root(ctk.CTk):

    # Class constructor
    def __init__(self):
        super().__init__()
        
        # ============= window settings =============
        self.title("XHand")
        ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
        
        # Setting the dimensions
        aspect_ratio = 9/16
        width = 1600
        height = round(width * aspect_ratio)
        self.geometry(f'{width}x{height}+0+0')
        
        self.notebook = NoteBook(container=self, width=width, height=height)
        self.notebook.pack()


class Controller():
    root = Root()
    root.mainloop()


if __name__ == '__main__':
    Controller()

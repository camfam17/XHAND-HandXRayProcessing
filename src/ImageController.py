import cv2 as cv
import os
import imageProcessing as ip

# Wrapper class to control the looping through selected images 
class ImageController():
    
    # Class constructor
    def __init__(self):
        self.images = []
    
    # Creates imageObjs from list of file paths provided
    def load_images(self, file_paths):
        for file in file_paths:
            self.images.append(ImageObj(file))
    
    # Returns the length of the images array
    def length(self):
        return len(self.images)
    
    # Returns the image object at the index provided
    def get_imageObj(self, index):
        return self.images[index]
    
    # Returns the filename 
    def get_file_name(self, index):
        return self.images[index].get_file_name()
    
    # Returns the original unprocessed image at the index
    def get_original_image(self, index):
        return self.images[index].get_original_image()
    
    # Returns the processed image object at the index
    def get_processed_image(self, index, stage):
        return self.images[index].get_processed_image(stage)
    
    # Returns the final processed image object at the index
    def get_final_processed_image(self, index):
        return self.images[index].get_processed_image(stage=6)
    
    # Get the Accepted/Rejected state of an image at the index
    def get_rejected(self, index):
        return self.images[index].get_rejected()
    
    # Toggle the state of the Rejected/Accepted status of the image object at the index.
    def toggle_rejected(self, index):
        self.images[index].toggle_rejected()
    
    # Saves the images currently in the images array.
    def save_images(self, save_path):
        for image in self.images:
            image.save_image(save_path)


class ImageObj():
    
    # Class constructor
    def __init__(self, file_path):
        
        self.file_path = file_path
        self.file_name = os.path.basename(self.file_path)
        
        self.rejected = False
    
    # Return the filename of the object
    def get_file_name(self):
        return self.file_name
    
    # Return the raw input image
    def get_original_image(self):
        
        if not hasattr(self, 'original_image'):
            self.original_image = cv.imread(self.file_path)
        
        return self.original_image
    
    # Process the image processed up to the desired stage and return it
    def get_processed_image(self, stage):
        
        # stage == -1: original
        # stage == 0: cropped
        # stage == 1: normalized
        # stage == 2: thresholded
        # stage == 3: annotated
        # stage == 4: verticle line
        # stage == 5: rotated
        # stage > 5: final (rotated, no lines or annotations)
        
        # currently reprocesses each image every time
        processed_image = ip.process_image(self.get_original_image(), stage=stage)
        
        return processed_image
    
    # Returns the final processed image of this object
    def get_final_processed_image(self):
        if not hasattr(self, 'final_processed_image'):
            self.final_processed_image = self.get_processed_image(stage=6)
        return self.final_processed_image
    
    # Saves this image, writing the final image to the desired path
    def save_image(self, save_path):
        
        if not self.rejected:
            status = cv.imwrite(save_path + '/' + self.file_name, self.get_final_processed_image())
            
            if not status:
                print('Error saving', self.file_name)
    
    # Get the Accepted/Rejected state of an image
    def get_rejected(self):
        return self.rejected
    
    # Toggles the image rejected state of this object
    def toggle_rejected(self):
        self.rejected = not self.rejected
    
    # Passes the image to be processed again in its final state to find features of the hand
    def get_hand_points(self): # (finger_tip_pinky, finger_tip_ring, finger_tip_middle, finger_tip_index, finger_tip_thumb) (palm_pinky_ring, palm_ring_middle, palm_middle_index, palm_index_thumb)
        if not hasattr(self, 'hand_points'):
            try:
                self.hand_points = ip.get_hand_points(self.get_final_processed_image())
            except:
                self.hand_points = ip.get_hand_points(ip.error_image)
        return self.hand_points
    
    # Calculates the area of triangles drawn between each finger as a representative of how splayed out the hand is, normalized against size of the image
    def get_splay_coefficient(self, *_):
        fingertips, palm_points = self.get_hand_points() # (finger_tip_pinky, finger_tip_ring, finger_tip_middle, finger_tip_index, finger_tip_thumb) (palm_pinky_ring, palm_ring_middle, palm_middle_index, palm_index_thumb)
        total_area = 0
        # Area =1/2[x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)]
        for i in range(len(fingertips)-1):
            f1 = fingertips[i]
            f2 = fingertips[i+1]
            p = palm_points[i]
            
            area = abs(0.5*( (f1[0] * (f2[1] - p[1])) + (f2[0] * (p[1] - f1[1])) + (p[0] * (f1[1] - f2[1])) ))
            total_area += area
        
        image_area = self.get_final_processed_image().size/100        
        return total_area/image_area
    
    
    # Calculates the size of the hand as the euclidean distance between pinky tip and thumb tip, normalized against size of the image
    def get_hand_size(self, *_):
        fingertips, _ = self.get_hand_points() # (finger_tip_pinky, finger_tip_ring, finger_tip_middle, finger_tip_index, finger_tip_thumb) (palm_pinky_ring, palm_ring_middle, palm_middle_index, palm_index_thumb)
        
        thumb_tip = fingertips[4]
        pinky_tip = fingertips[0]
        
        img = self.get_final_processed_image()
        img = ip.draw_points(img, (thumb_tip, pinky_tip))
        
        return ip.euclidean_distance(thumb_tip, pinky_tip)/self.get_final_processed_image().size * 10000
    
    
    # Calculates the outline contours of the final processed hand
    def get_contours(self, *_):
        
        if not hasattr(self, 'contours'):
            component_image, _ = ip.component_analysis(ip.process_image(self.get_final_processed_image(), stage=2))
            
            contours1 = cv.findContours(component_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
            contours2 = contours1[0]
            self.contours = []
            for i in contours2:
                self.contours.append(i[0])
        
        return self.contours
    
    # Calculates the alignment error between the contours of this image with the target image
    def get_contour_alignment_error(self, target_image, option):
        self.contour_alignment_error = ip.calc_alignment_error(self.get_contours(), target_image.get_contours(), option)
        return self.contour_alignment_error
    
    # Calculates the alignment error between the fingertips of this image with the target image
    def get_fingertip_alignment_error(self, target_image, option):
        self.fingertip_alignment_error = ip.calc_alignment_error(self.get_hand_points()[0], target_image.get_hand_points()[0], option)
        return self.fingertip_alignment_error
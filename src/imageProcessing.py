from tkinter import filedialog
import cv2
import numpy as np
import math
import imutils

error_image = cv2.imread('errorimg.png', 1) 

# Crop image by desired value while keeping image centered
def center_crop(input_image, crop):
    
    image = input_image.copy()
    
    width, height = image.shape[1], image.shape[0]
    
    # process crop width and height for max available dimension
    crop_width = image.shape[1]*crop 
    crop_height = image.shape[0]*crop
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    cropImg = image[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return cropImg

# Converts an image to grayscale then slightly blurs it before performing normalization
def normalize_image(cropped_input_image):
    
    image = cropped_input_image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3)) # blur the image reduces noise
    hst = cv2.normalize(blur, None, 0, 1800, cv2.NORM_MINMAX)
    
    return hst

# Thresholds a normalized image based on the supplies thresholding value
def threshold_image(normalized_image, threshold):
    
    image = normalized_image.copy()
    
    ret, threshHistImage = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return threshHistImage

# Connected Component Analysis returns image of component and the size of the largest component
def component_analysis(thresholded_image):
    
    image = thresholded_image.copy()
    
    num_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]
    
    max_label = 1
    max_size = sizes[1]
    for i in range(2, num_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    
    #create black image for the largest component
    component_image = np.zeros(output.shape)
    component_image= (output == max_label).astype("uint8")*255

    # cv2.imshow("DILATED", dilation)
    # cv2.imshow("NORMAL", component_image)
    # cv2.waitKey()
    return component_image, max_size


# # Removes background components from the image by extracting the hand component and dilating it to prevent detail lose
def remove_clutter(thresholded_imageX, raw_imageX):
    
    thresholded_image = thresholded_imageX.copy()
    thresholded_image = cv2.resize(thresholded_image, None, fx=0.25, fy=0.25) # Down-scale for efficiency
    raw_image = raw_imageX.copy()
    
    num_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresholded_image, connectivity=4)
    sizes = stats[:, -1]
    
    max_label = 1
    max_size = sizes[1]
    for i in range(2, num_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    #create black image for the largest component
    component_image = np.zeros(output.shape)
    component_image= (output == max_label).astype("uint8")*255

    #Prepare to dialate largest component
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(component_image,kernel,iterations = 3)
    

    #Convert mask to RGB image
    components_to_mask_3_channel = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)

    #Change all black pixels in mask to match background value
    indices = np.where(components_to_mask_3_channel==0)
    components_to_mask_3_channel[indices[0], indices[1], :] = [6, 6, 6]
    components_to_mask_3_channel = cv2.resize(components_to_mask_3_channel, (raw_image.shape[1], raw_image.shape[0]))
    
    altered = cv2.bitwise_and(raw_image, components_to_mask_3_channel)
    
    return altered


# Find the midpoint between 2 coordinates
def midpoint(p1, p2):
    return round((p1[0]+p2[0])/2), round((p1[1]+p2[1])/2)


# Finds the contours of the image and marks key points returns, marks convexity defects and returns lists of palm points and finger tips 
def contour_analysis(component_image, cropped_input_image, dist_threshold):
    
    image = component_image.copy()
    draw_image = cropped_input_image.copy()
    
    convexhull = []
    palm_points = []
    fingertips = []
    fingertips_end = []
    #RETR_TREE and RETR_EXTERNAL seem to produce same results
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(draw_image, contours, -1, (0,255,75), 1)
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        convexhull.append(cv2.convexHull(contours[i], False))
    
    # draw contours and hull points
    for i in range(len(contours)):
        #Check for only largest contour
        if (cv2.contourArea(contours[i]) > 1000):
            # draw ith contour
            cv2.drawContours(draw_image, convexhull, i, [255,0,0], 1, 8)
            # draw ith convex hull object
            hull = cv2.convexHull(contours[i], returnPoints=False)
            convexityDefects = cv2.convexityDefects(contours[i], hull)
            for z in range(convexityDefects.shape[0]):
                s, e, furthest, dist = convexityDefects[z, 0]
                start = tuple(contours[i][s][0])
                end = tuple(contours[i][e][0])
                palm = tuple(contours[i][furthest][0])
                if (dist > dist_threshold):
                    cv2.circle(draw_image, start, 9, [255,0,0], -1)
                    cv2.circle(draw_image, palm, 9, [0,0,255], -1)
                    cv2.circle(draw_image, end, 9, [255,0,255], -1)
                    palm_points.append(palm)
                    fingertips.append(start)
                    fingertips_end.append(end)
    return draw_image, palm_points, fingertips, fingertips_end


# Mark the middle line on the middle finger if possible, will return an error if not
def drawLineFinger(cropped_input_image, size, palm_points, fingertips, fingertips_end):
    #if (len(palm_points) == 6): # rudimentary test for successful contour
    drawImage = cropped_input_image.copy()
    try:
        finger_base = midpoint(palm_points[0],palm_points[len(palm_points)-1])
        finger_tip = midpoint(fingertips[0],fingertips_end[len(fingertips_end)-1])
        cv2.circle(drawImage, finger_base, 9, [0,0,255], -1)
        cv2.circle(drawImage, finger_tip, 9, [0,0,255], -1)
        cv2.line(drawImage, finger_base, finger_tip, [255,0,255], 2)

        return drawImage, finger_base, finger_tip
    except Exception as e:
        print("IMAGE NOT CORRECTLY PROCESSED")
        print("PALM POINTS: ", len(palm_points))
        print("COMPONENT SIZE: ", size)
        return error_image, 0, 0

#Rotates the image to align the middle finger vertically
def rotate(image, finger_base, finger_tip):
    image_copy = image.copy()

    dx = finger_tip[0] - finger_base[0]
    dy = -(finger_tip[1] - finger_base[1])
    
    alpha = math.degrees(math.atan2(dy, dx))
    rotation = 90-alpha
    
    rotated_img = imutils.rotate(image_copy, rotation)
    return rotated_img

#translates the middle finger to center on a point in the middle of the palm
def translate_image(lined_image, finger_base):
    image = lined_image.copy()
    
    height, width = image.shape[:2]
    tx, ty = (width/2)-finger_base[0],  (height/2) - finger_base[1]  
    ty = ty - height*0.1
    
    translation_matrix = np.array([[1, 0, tx],[0, 1, ty]], dtype=np.float32) 
    translated_image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(width, height))
    return translated_image

# Translates then rotates the image for the middle finger to be vertical
def transform_image(lined_image, finger_base, finger_tip):
    translated_image = translate_image(lined_image, finger_base)
    rotated_img = rotate(translated_image, finger_base, finger_tip)
    return rotated_img


# Overlays the image and template over each other
def overlay_images(image, template):
    
    # Resizes the images to have the same dimensions
    image = cv2.resize(image, (500, 500))
    template = cv2.resize(template, (500, 500))
    
    # Convert to binary
    image_msk = cv2.threshold(image, 14, 255, cv2.THRESH_BINARY)[1]  # IDEAL THRESH: 16
    image2_msk = cv2.threshold(template, 14, 255, cv2.THRESH_BINARY)[1]
    
    # ANDs an image with its own mask
    image_mask = cv2.bitwise_and(image, image_msk)
    template_mask = cv2.bitwise_and(template, image2_msk)
    
    # Apply colormap
    image = cv2.applyColorMap(image_mask, cv2.COLORMAP_BONE)
    template = cv2.applyColorMap(template_mask, cv2.COLORMAP_PINK)
    
    # overlays image on top of template
    alpha = 1
    overlay = cv2.addWeighted(image, alpha, template, beta=0.5, gamma=0)
    
    return overlay

# Displays the suplied image with whatever title provided - used in testing
def display(title, image):
    image = cv2.resize(image, None, fx=0.25, fy=0.25)
    cv2.imshow(title, image)

# Draws red circles on the image provided based on coordinates from a points array - used in testing
def draw_points(imageX, points):
    
    image = imageX.copy()
    
    for point in points:
        cv2.circle(image, point, 9, [0,0,255], -1)
    
    return image

# Finds and retuns all signficant finger tips and palm points for use in feature analysis
def get_hand_points(image):
    
    threshold = 170
    dist_threshold = 5750
    
    input_image = image.copy()
    
    normalized_image = normalize_image(input_image)
    
    thresholded_image = threshold_image(normalized_image, threshold)
    
    component_image, size = component_analysis(thresholded_image)
    
    outline_image, palm_points, fingertips, fingertips_end = contour_analysis(component_image, input_image, dist_threshold)
    
    finger_tip_index = midpoint(fingertips[len(fingertips)-1],fingertips_end[len(fingertips_end)-2])
    finger_tip_middle = midpoint(fingertips[0],fingertips_end[len(fingertips_end)-1])
    finger_tip_ring = midpoint(fingertips[1],fingertips_end[0])
    finger_tip_pinky = midpoint(fingertips[2],fingertips_end[1])
    finger_tip_thumb = midpoint(fingertips[len(fingertips)-2],fingertips_end[len(fingertips_end)-3])
    
    if finger_tip_pinky == finger_tip_thumb:
        finger_tip_pinky = fingertips_end[1]
        finger_tip_thumb = fingertips[-2]
    
    palm_pinky_ring = palm_points[1]
    palm_ring_middle = palm_points[0]
    palm_middle_index = palm_points[-1]
    palm_index_thumb = palm_points[-2]
    
    finger_tips = (finger_tip_pinky, finger_tip_ring, finger_tip_middle, finger_tip_index, finger_tip_thumb)
    palm_point = (palm_pinky_ring, palm_ring_middle, palm_middle_index, palm_index_thumb)
    
    return finger_tips, palm_point

# Returns euclidean distance between two points
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


# Calculates the alignment error between 2 sets of points using sum of squared differences (ssd), returns sum, mean or standev of ssd
def calc_alignment_error(image1_contours, image2_contours, option):
    distance_vector = []
    
    min_contour = min((image1_contours, image2_contours), key=lambda x: len(x))
    
    for i in range(len(min_contour)):
        distance = ((image1_contours[i][0] - image2_contours[i][0]) ** 2 + (image1_contours[i][1] - image2_contours[i][1]) ** 2) ** 0.5
        distance_vector.append(distance)
    
    sum = 0
    for i in range(len(distance_vector)):
        sum = sum + distance_vector[i]  # Sums all the distance of each point
    
    if option == 'sum':
        return int(sum)
    
    mean_error = sum/len(distance_vector)
    
    if option == 'mean':
        return int(mean_error)
    
    summation = 0
    for i in range(len(distance_vector)):
        summation = summation + (distance_vector[i] - mean_error)**2
    
    n = len(distance_vector)           # Size of sample
    standard_dev = (summation/(n - 1))**0.5
    return int(standard_dev)


# Pipeline of processing images
def process_image(input, stage=6, do_remove_clutter=True):
    
    threshold = 170
    crop = 0.96
    dist_threshold = 5750
    
    # stage == -1: original
    # stage == 0: cropped
    # stage == 1: normalized
    # stage == 2: thresholded
    # stage == 3: annotated
    # stage == 4: verticle line
    # stage == 5: rotated
    # stage > 5: final (rotated, no lines or annotations)
    
    # Detects if image or file-path provided as input
    if type(input) == str:
        input_image = cv2.imread(input, 1)
    else:
        input_image = input.copy()
    
    if stage == -1:
        return input_image
    
    cropped_input_image = center_crop(input_image, crop)#
    if stage == 0:
        return cropped_input_image
    
    normalized_image = normalize_image(cropped_input_image)
    if stage == 1:
        return normalized_image
    
    thresholded_image = threshold_image(normalized_image, threshold)
    if stage == 2:
        return thresholded_image
    
    try:
        component_image, size = component_analysis(thresholded_image)#
        outline_image, palm_points, fingertips, fingertips_end = contour_analysis(component_image, cropped_input_image, dist_threshold)#
        
        if stage == 3:
            return outline_image
        
        lined_image, finger_base, finger_tip = drawLineFinger(cropped_input_image, size, palm_points, fingertips, fingertips_end)#
        if stage == 4:
            return lined_image
        
        rotated_lined_image = transform_image(lined_image, finger_base, finger_tip)
        if stage == 5:
            return rotated_lined_image
        
        
        if do_remove_clutter:
            cropped_input_image = remove_clutter(thresholded_image, cropped_input_image)
        
        
        final_image = transform_image(cropped_input_image, finger_base, finger_tip)
        return final_image
    except:
        return error_image


# Only used when testing
if __name__ == '__main__':
    
    global img1
    img1 = cv2.imread(filedialog.askopenfilename(), 1)
    
    display('final', process_image(img1, do_remove_clutter=False))
    display('outline', process_image(img1, stage=3))
    get_hand_points(img1)
    
    print('done')
    
    cv2.waitKey(0)

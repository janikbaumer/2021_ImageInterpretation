#A function to perform Canny edge detection using the opencv library
def edge_detection(input_image, threshold1, threshold2):
    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges

#Function call example
output_img = edge_detection(input_img, 50, 100) #e.g. Threshold1 = 50, Threshold2 = 100  

#A function which computes the average intensity in the neighbourhood around a pixel (has a blurring, smoothing effect)
def nearest_neighbour_average(input_image, kernel_size):
    #Convert to grayscale to obtain a signal channel
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    #Define a kernel (ideally an odd size so that there are neighbours all around)
    kernel = np.ones((kernel_size, kernel_size))
    kernel[int(np.floor(kernel_size/2)), int(np.floor(kernel_size/2))] = 0; #Centre elements of the Kernel
    print(kernel.ndim)
    print(type(kernel))
    output_image = signal.convolve2d(grayscale, kernel, boundary='pad', mode='same')/kernel.sum()  #Don't forget to normalize by sum of kernel elements
    #Note: At the edges, the function pads the kernel with zeros
    return output_image

#Function call example
output_img = nearest_neighbour_average(input_img, 31) #e.g. Compute the average over a 31 x 31 kernel

#A window-level transfer function which stretches and manipulates intensities to enhance contrast, has an effect similar to colour segmentation
def window_level_function(image, window, level):
    #Convert to grayscale to obtain a signal channel
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.double)  #Convert the input image into double format
    minimum = np.min(image[:]) #Minimum intensity value in the image
    maximum = np.max(image[:]) #Maximum intensity value in the image
    
    #Clip or remove everything before or after the window boundaries
    image = np.clip(image, (level-(window/2)), (level+(window/2))) 
    #Everything below left limit becomes black
    #Everyting above right limit becomes white
    
    m = (maximum-minimum)/window; #Slope of the window level transfer function
    b = maximum - (m * (level + (window/2))) #y-intercept of the window level transfer function

    image = m * image + b #The remaining, non-scaled values are adjusted by applying a linear transformation
    return image.astype(np.uint8) #Convert output to unsigned 8-bit integer and return

#Function call example
output_img = window_level_function(input_img, 10, 250) #e.g. window = 10, level = 250

#A function that introduces some blurring from a Gaussian filter
def gaussian_filter(input_image, sigma, kernel_size): 
    #Convert to grayscale to obtain a signal channel
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gauss_kernel = np.zeros((kernel_size, kernel_size)) #Declare an empty matrix of zeros based on the kernel size
    indices = [*range(-int(np.floor(kernel_size/2)), int(np.floor(kernel_size/2))+1)] #Get the neighbouring indices
    constant = 1/(2*(np.pi)*sigma**2) #Constant term in the above equation
    
    #For all indices in both dimensions
    for i in range(kernel_size):
        for j in range(kernel_size):
            #Evaluate the 2D Gaussian function to obtain the Kernel
            gauss_kernel[i, j] = constant*(np.exp(-((indices[i])**2+(indices[j])**2)/(2*sigma**2)))
        
    gauss_kernel = gauss_kernel / np.sum(gauss_kernel.ravel()) #Normalize the kernel by dividing by the sum of elements
    
    output_image = signal.convolve2d(grayscale, gauss_kernel, boundary='pad', mode='same')
    return output_image    

#Function call example
output_img = gaussian_filter(input_img, 5, 11) #e.g. standard deviation = 5, using an 11 x 11 kernel

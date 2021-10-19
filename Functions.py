#Window-level function to manipulate image histogram
def window_level_function(image, window, level): #Input an image, a window, and a level
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


#Gaussian filter/blur function
def gaussian_kernel(sigma, kernel_size): #Input standard deviation and kernel (larger standard deviation and larger kernel results in more blur, but this takes more time to run)   
    gauss_kernel = np.zeros((kernel_size, kernel_size)) #Declare an empty matrix of zeros based on the kernel size
    indices = [*range(-int(np.floor(kernel_size/2)), int(np.floor(kernel_size/2))+1)] #Get the neighbouring indices
    constant = 1/(2*(np.pi)*sigma**2) #Scale factor
    
    #For all indices in both dimensions
    for i in range(kernel_size):
        for j in range(kernel_size):
            #Evaluate the 2D Gaussian function to obtain the Kernel
            gauss_kernel[i, j] = constant*(np.exp(-((indices[i])**2+(indices[j])**2)/(2*sigma**2)))
        
    gauss_kernel = gauss_kernel / np.sum(gauss_kernel.ravel()) #Normalize the kernel by dividing by the sum of elements
    
    return gauss_kernel
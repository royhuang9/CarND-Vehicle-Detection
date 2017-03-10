import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
            

def show_updown(image1, image2):
    fig = plt.figure(figsize=(12,9))
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1,1],
                           height_ratios=[1,1]
                           )
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    ax1.imshow(image1, cmap='Greys_r')
    #ax1.axis('off')
    #ax1.set_title('')
    ax1.set_xlim(0, image1.shape[1])
    ax1.set_ylim(image1.shape[0], 0)
    
    ax2.imshow(image2, cmap='Greys_r')
    #ax2.axis('off')
    #ax2.set_title('Bird view')
    ax2.set_xlim(0, image2.shape[1])
    ax2.set_ylim(image2.shape[0], 0)
    
    fig.tight_layout()
    
def show_gray(image, title=''):
    plt.figure(figsize=(12,9))
    plt.imshow(image, cmap='gray')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    if title:
        plt.title(title)
    plt.show()
    

def show_overlay(image, loc, ylmt=(0,255)):
    h, w = image.shape[:2]
    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(111)
    ax1.imshow(image, 'gray')
    ax1.plot([0, w], [loc, loc],'b', linewidth=4)
    ax1.set_xlim(0, w)
    ax1.set_ylim(h, 0)

    ax2 = ax1.twinx()
    ax2.plot(image[loc, :], 'r')
    #ax2.set_ylabel('y2', color='r')
    ax2.set_ylim(ylmt[0], ylmt[1])
    ax2.set_xlim(0, w)
    plt.show()

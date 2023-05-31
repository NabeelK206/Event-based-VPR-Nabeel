
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.animation as PillowWriter 
import math
import scipy.signal
from numpy import sin,cos,pi
from scipy.spatial.distance import cdist 
from scipy.integrate import cumtrapz


# PARAMETERS
seq_len=1
sensor_size = (346, 260)
filename_ref='event_array_ref.npy'
filename_query='event_array_qry.npy'
match=0


def sequenceVRPwithSlicedEventImages(filename_ref,filename_query, seq_len):

    with open(filename_ref, 'rb') as f:
        events_frame_ref = np.load(f)
    with open(filename_query, 'rb') as f:
        events_frame_query = np.load(f)

    print(len(events_frame_query))
    print(len(events_frame_ref))
    time_diff=abs(len(events_frame_ref)-len(events_frame_query))

    feats_ref=np.zeros((len(events_frame_ref)-time_diff,sensor_size[0]*sensor_size[1]))
    feats_qry=np.zeros((len(events_frame_ref)-time_diff,sensor_size[0]*sensor_size[1]))
    for i in range(0,len(events_frame_ref)-time_diff):
        feats_ref[i,:]=(events_frame_ref[i].flatten())
        feats_qry[i,:]=(events_frame_query[i].flatten())
    print("Done flattening and joining frames")

    dMat = cdist(feats_ref,feats_qry,'euclidean')
    print("Done distance matrix")
    

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(dMat, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()
    
    from scipy import signal
    convdMat=np.array(signal.convolve2d(dMat, np.identity(seq_len,dtype=int), mode='valid'))
    mIndSeqs = np.argmin(convdMat,axis=0)
    print("Done compiling indexes for sequence matching")


    global match

    for i in range(0,len(events_frame_ref)-time_diff):
        print(" Match: '" +str(mIndSeqs[i]) + " time: '" +str(i))
        if abs(mIndSeqs[i]-i)<5:
            match=match+1
    
    print(match)
    

    blank=np.zeros((sensor_size[1],sensor_size[0]))
    fig4, (ax1, ax2, ax3)= plt.subplots(1,3)
    fig4.set_tight_layout(True)
    ax1.imshow(blank, animated=True)
    ax2.imshow(blank, animated=True)
    ax3.imshow(blank, animated=True)

    def animate(i):
        qry=events_frame_query[i]
        gndt=events_frame_ref[i]
        match=events_frame_ref[mIndSeqs[i]]
        ax1.set_title("Query Event: "+ str(i))
        ax2.set_title( " Match: " +str(mIndSeqs[i]))
        ax3.set_title("Ground Truth")

        ax1.imshow(qry, animated=True,cmap='twilight_shifted')
        ax2.imshow(match, animated=True,cmap='twilight_shifted')
        ax3.imshow(gndt, animated=True,cmap='twilight_shifted')

    anim = animation.FuncAnimation(fig4, animate, frames=len(events_frame_ref)-time_diff, interval = 20)
    fig4.suptitle('Event Capture', fontsize=14)


    writervideo = animation.FFMpegWriter(fps=1) 
    anim.save("images.mp4", writer=writervideo)

    plt.show()

    
    




'''Test'''
sequenceVRPwithSlicedEventImages(filename_ref,filename_query, seq_len)



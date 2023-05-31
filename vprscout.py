import rospy
import numpy as np
from scipy.signal import medfilt2d
import matplotlib.pyplot as plt 
from dvs_msgs.msg import EventArray
import cv2
from cv_bridge import CvBridge, CvBridgeError
from scipy import signal
from scipy.spatial.distance import cdist 
import time

from sensor_msgs.msg import Image
from matplotlib.pyplot import cm



x_list = []
p_list = []
y_list = []
t_list = []
events_frame_qry=[]
sensor_size=(346, 260)
counter=0
match=0

events_frame_ref = np.load('event_array_ref.npy')
feats_ref=np.zeros((len(events_frame_ref),sensor_size[0]*sensor_size[1]))
for i in range(0,len(events_frame_ref)):
    feats_ref[i,:]=(events_frame_ref[i].flatten())


def callback(data):
    global t_list, x_list, y_list, p_list
    for i in range(len(data.events)):
       p_list.append(int(data.events[i].polarity))
       x_list.append(data.events[i].x)
       y_list.append(data.events[i].y)
       t_list.append(data.events[i].ts.to_sec())
         

    

def timer_callback(timer_event):

    global t_list, x_list, y_list, p_list, events_frame_qry, counter,match

    counter = counter + 1
     
    #time duration of callback fucntion 
    t_start = time.time()
    t_np = np.array(t_list)

    if len(t_list)>0: 

        last_timestamp=(np.array(t_list)[-1])
        idx= [x for x, val in enumerate(t_np) if ((val > last_timestamp-0.1) & (val < last_timestamp))]

        if len(idx)>0:
            x_np_slice = np.array(x_list)[idx]
            y_np_slice = np.array(y_list)[idx]
            p_np_slice = np.array(p_list)[idx]

            events_frame=np.zeros((sensor_size[1],sensor_size[0]))+255

            for i in range(len(p_np_slice)):
                events_frame[y_np_slice[i], x_np_slice[i]]=p_np_slice[i]
                
                
            event_frame_filter=medfilt2d(events_frame, 3)
            events_frame_qry.append(event_frame_filter)
            
            
            print(len(events_frame_qry))

            seq_len=1

            if len(events_frame_qry)>seq_len:
                event_frame_qry=events_frame_qry[-seq_len:]
                #flattening query image 
                feats_qry=np.zeros((seq_len,sensor_size[0]*sensor_size[1]))
                events_frame_query_seq=events_frame_qry[-seq_len:]
                #print(len(events_frame_query_seq))
                for i in range(seq_len):    
                    feats_qry[i,:]=events_frame_query_seq[i].flatten()

                
                #print(feats_qry)
                #print(feats_ref)

                #sequence matching 	
                dMat = cdist(feats_ref,feats_qry,'euclidean')

                #print(dMat)
                convdMat=np.array(signal.convolve2d(dMat,np.identity(seq_len,dtype=int), mode='valid'))
                mIndSeqs = np.argmin(convdMat,axis=0)

                #print(len(mIndSeqs))
                #print(len(events_frame_ref))

                

                img = np.uint8((np.array(events_frame_ref[mIndSeqs[-1]])/2)*128)
                imC = cv2.applyColorMap(img, cv2.COLORMAP_TWILIGHT_SHIFTED)

                img_gt = np.uint8((np.array(events_frame_ref[counter])/2)*128)
                imC_gt = cv2.applyColorMap(img_gt, cv2.COLORMAP_TWILIGHT_SHIFTED)

                

                if abs(mIndSeqs-counter)<3:
                    match=match+1


                if len(events_frame_qry) <= len(events_frame_ref):
                    np.save('event_array_qry', events_frame_qry)


                #print(match)    
                # print(" index: '" +str(mIndSeqs) + " time: '" +str(counter))
                # print("matches '" +str(match)+ " time: '" +str(counter))

                font = cv2.FONT_HERSHEY_SIMPLEX  
                org = (20, 20)
                fontScale = 0.75
                color = (0, 0, 0)
                thickness = 2

                text = "    Qry Event: [" + str(counter)+ "]                Ref Match: "+str(mIndSeqs)+"            Ground Truth: [" +str(counter)+"]"

                img_q = np.uint8((np.array(event_frame_filter)/2)*128)
                imC_q = cv2.applyColorMap(img_q, cv2.COLORMAP_TWILIGHT_SHIFTED)

                h_img= cv2.hconcat([imC_q, imC,imC_gt])

                cv2.namedWindow('Events',cv2.WINDOW_NORMAL) 

                cv2.resizeWindow('Events', 1920,1080)
                cv2.putText(h_img,text,org,font,fontScale,color,thickness)
                cv2.imshow('Events',h_img)
                
                cv2.waitKey(1)
            





                

            p_list = p_list[np.min(idx):]
            x_list = x_list[np.min(idx):]
            y_list = y_list[np.min(idx):]
            t_list = t_list[np.min(idx):]
            
            #print('ref length')
            

            
                
    end_time = time.time()
    ##print('Time: %.2f' % (end_time-t_start))	
            









def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Timer(rospy.Duration(1), timer_callback)
    rospy.Subscriber('/dvs/events', EventArray, callback)


    rospy.spin()

if __name__ == '__main__':
    listener()
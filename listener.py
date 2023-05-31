import rospy
import numpy as np
from dvs_msgs.msg import EventArray
import time
from scipy.signal import medfilt2d

x_list = []
p_list = []
y_list = []
t_list = []
events_frame_ref=[]
sensor_size=(346, 260)

def callback(data):
    global t_list, x_list, y_list, p_list
    for i in range(len(data.events)):
       p_list.append(int(data.events[i].polarity))
       x_list.append(data.events[i].x)
       y_list.append(data.events[i].y)
       t_list.append(data.events[i].ts.to_sec())

    

def timer_callback(timer_event):

    global t_list, x_list, y_list, p_list, events_frame_ref
    
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
            events_frame_ref.append(event_frame_filter)

            np.save('event_array_ref', events_frame_ref)
            print(len(events_frame_ref))
            
            p_list = p_list[np.min(idx):]
            x_list = x_list[np.min(idx):]
            y_list = y_list[np.min(idx):]
            t_list = t_list[np.min(idx):]
            
            
    	
            
                
    


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Timer(rospy.Duration(1), timer_callback)
    rospy.Subscriber('/dvs/events', EventArray, callback)
    rospy.spin()


    

if __name__ == '__main__':
    listener()
    
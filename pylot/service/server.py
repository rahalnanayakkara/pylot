import socket

import pickle
import threading

import tracker
import service

from controller import get_control_message
from tracker import get_obstacle_tracker_message

def controller_server():
    host = "172.31.9.13"
    port = 5010

    server_socket = socket.socket() 
    server_socket.bind((host, port))

    # configure how many client the server can listen simultaneously
    server_socket.listen(10)
    conn, address = server_socket.accept()  

    while True:
        #print("Connection from: " + str(address))
        # conn, address = server_socket.accept() 
        print("Connection from: " + str(address))
        #input_string = b""
        
        #while True:
            # receive data stream. it won't accept data packet greater than 1024 bytes
        #    packet = conn.recv(4096)
        #    if len(packet) == 0: break
        #    input_string += packet
        
        #if not input_string:
        # if data is not received break
        #    break
        input_string = service.recv_msg(conn)

        # print(len(input_string))
        # print(input_string[:100])
        # print(input_string[-100:])
        
        input = pickle.loads(input_string)
        print("Received control input: ", len(input_string))
        control_message = get_control_message(
            pose=input.pose, 
            waypoints=input.waypoints, 
            type=input.type
            )

        print("Generated control message: ", control_message)
        conn.send(pickle.dumps(control_message))  # send data to the client
    
    conn.close()  # close the connection

    


def tracker_server():
    host = "172.31.9.13"
    port = 5020

    server_socket = socket.socket() 
    server_socket.bind((host, port))

    # configure how many client the server can listen simultaneously
    server_socket.listen(10)
    conn, address = server_socket.accept()  
    sort_tracker = tracker.MultiObjectSORTTracker()
    deepsort_tracker = tracker.MultiObjectDeepSORTTracker()

    while True:
        # receive data stream. it won't accept data packet greater than 1024 bytes
        # input_message = b""
        # while len(input_message) < 1772060:
        #    packet = conn.recv(4096)
        #    if not packet: break
        #    input_message += packet
        
        input_message = service.recv_msg(conn)
        input = pickle.loads(input_message)
        if not input:
            print("NO input!!!!!!!!!!!!")
        else:
            print("Received tracker input: ", len(input_message))
        if input.type=="sort":
            print("SORT")
            tracker_message = get_obstacle_tracker_message(
                frame=input.frame,
                obstacles=input.obstacles,
                reinit=input.reinit,
                tracker=sort_tracker
            )
        else:
            tracker_message = get_obstacle_tracker_message(
                frame=input.frame,
                obstacles=input.obstacles,
                 reinit=input.reinit,
               tracker=deepsort_tracker
            )
        print("Generated tracker message: ", tracker_message)
        conn.send(pickle.dumps(tracker_message))  # send data to the client
    
    conn.close()  # close the connection


if __name__=='__main__':
    thread_one = threading.Thread(target=controller_server)
    thread_two = threading.Thread(target=tracker_server)
    thread_one.start()
    thread_two.start()
    #tracker_server()

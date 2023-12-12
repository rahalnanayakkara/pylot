import socket

import pickle
import threading

from controller import get_control_message
from tracker import get_obstacle_tracker_message

def controller_server():
    host = "0.0.0.0"
    port = 5001

    server_socket = socket.socket() 
    server_socket.bind((host, port))

    # configure how many client the server can listen simultaneously
    server_socket.listen(10)
    conn, address = server_socket.accept()  

    print("Connection from: " + str(address))
    while True:
        # receive data stream. it won't accept data packet greater than 1024 bytes
        input_message = conn.recv(102400)
        if not input_message:
            # if data is not received break
            break
        
        input = pickle.loads(input_message)
        print("Received input message: ", input)
        control_message = get_control_message(
            pose=input.pose, 
            waypoints=input.waypoints, 
            type=input.type
            )

        print("Generated control message: ", control_message)
        conn.send(pickle.dumps(control_message))  # send data to the client
    conn.close()  # close the connection


def tracker_server():
    host = "0.0.0.0"
    port = 5002

    server_socket = socket.socket() 
    server_socket.bind((host, port))

    # configure how many client the server can listen simultaneously
    server_socket.listen(10)
    conn, address = server_socket.accept()  

    while True:
        # receive data stream. it won't accept data packet greater than 1024 bytes
        input_message = conn.recv(512000)
        if not input_message:
            # if data is not received break
            break
        
        input = pickle.loads(input_message)
        print("Received input message: ", input)
        tracker_message = get_obstacle_tracker_message(
            frame=input.frame,
            obstacles=input.obstacles,
            reinit=input.reinit,
            type=input.type
        )

        print("Generated tracker message: ", tracker_message)
        conn.send(pickle.dumps(tracker_message))  # send data to the client
    conn.close()  # close the connection


if __name__=='__main__':
    thread_one = threading.Thread(target=controller_server)
    thread_two = threading.Thread(target=tracker_server)
    thread_two.start()

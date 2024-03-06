import signal

# import carla_wrapper.utils

simulator_host = 'localhost'
simulator_port = 2000
simulator_timeout = 10
carla_traffic_manager_port = 8000

def get_world(host: str = "localhost", port: int = 2000, timeout: int = 10):
    """Get a handle to the world running inside the simulation.

    Args:
        host (:obj:`str`): The host where the simulator is running.
        port (:obj:`int`): The port to connect to at the given host.
        timeout (:obj:`int`): The timeout of the connection (in seconds).

    Returns:
        A tuple of `(client, world)` where the `client` is a connection to the
        simulator and `world` is a handle to the world running inside the
        simulation at the host:port.
    """
    try:
        from carla import Client
        client = Client(host, port)
        client_version = client.get_client_version()
        server_version = client.get_server_version()
        err_msg = 'Simulator client {} does not match server {}'.format(
            client_version, server_version)
        assert client_version == server_version, err_msg
        client.set_timeout(timeout)
        world = client.get_world()
    except RuntimeError as r:
        raise Exception("Received an error while connecting to the "
                        "simulator: {}".format(r))
    except ImportError:
        raise Exception('Error importing CARLA.')
    return (client, world)

# def shutdown_sim(node_handle, client, world):
#     if node_handle:
#         node_handle.shutdown()
#     else:
#         print('WARNING: The Pylot dataflow failed to initialize.')
#     set_asynchronous_mode(world)
#     if pylot.flags.must_visualize():
#         import pygame
#         pygame.quit()


# def shutdown(sig, frame):
#     raise KeyboardInterrupt

def main():
    # Connect an instance to the simulator to make sure that we can turn the
    # synchronous mode off after the script finishes running.
    client, world = get_world(simulator_host, simulator_port,
                              simulator_timeout)
    # try:
    #     # node_handle, control_display_stream = driver()
    #     signal.signal(signal.SIGINT, shutdown)
    #     node_handle.wait()
    # except KeyboardInterrupt:
    #     shutdown_sim(node_handle, client, world)
    # except Exception:
    #     shutdown_sim(node_handle, client, world)
    #     raise


if __name__ == '__main__':
    main()
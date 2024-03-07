import params

from objects.frames import CameraFrame
from objects.objects import Transform, RGBCameraSetup, Rotation, Location

# The location of the center camera relative to the ego-vehicle.
CENTER_CAMERA_LOCATION = Location(1.3, 0.0, 1.8)

transform = Transform(CENTER_CAMERA_LOCATION, Rotation(pitch=-15))
camera_setup = RGBCameraSetup('center_camera',
                                    params.camera_image_width,
                                    params.camera_image_height, transform,
                                    params.camera_fov)

class CarlaCamera:
    
    def __init__(self, world, vehicle):

        # Install the camera.
        camera_blueprint = world.get_blueprint_library().find(camera_setup.camera_type)
        camera_blueprint.set_attribute('image_size_x', str(camera_setup.width))
        camera_blueprint.set_attribute('image_size_y', str(camera_setup.height))
        camera_blueprint.set_attribute('fov', str(camera_setup.fov))
        
        if params.simulator_camera_frequency == -1:
            camera_blueprint.set_attribute('sensor_tick', '0.0')
        else:
            camera_blueprint.set_attribute('sensor_tick', str(1.0 / params.simulator_camera_frequency))

        transform = camera_setup.get_transform().as_simulator_transform()

        self._camera = world.spawn_actor(camera_blueprint,
                                         transform,
                                         attach_to=vehicle)

        self._processed_images = {}

        # Register the callback on the camera.
        self._camera.listen(self.process_images)

    def process_images(self, simulator_image):
        """Invoked when an image is received from the simulator."""
        game_time = int(simulator_image.timestamp * 1000)
        if self._camera_setup.camera_type == 'sensor.camera.rgb':
            frame = CameraFrame.from_simulator_frame(simulator_image, self._camera_setup)
        self._processed_images[game_time] = frame
    
    def get_processed_image(self, game_time):
        return self._processed_images[game_time]

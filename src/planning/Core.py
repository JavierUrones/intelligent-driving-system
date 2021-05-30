import src.robot.RobotManager as rm
import src.cvision.VisionManager as vm

class Core:
    def __init__(self):
        self.robot = rm.RobotManager
        self.vision = vm.VisionManager

    def start_autonomous_driving(self):
        #Obtener imagenes del vision manager
        #Obtener valor del sensor
        #Si hay algo delante del sensor parar siempre.
        #Si no interpretar imagen y predecir con modelo inteligente.
        #Actuar conforme a lo que prediga el modelo inteligente.
        print("empty")
        steering_angle = 0
        while True:
            steering_angle = self.lane_following(steering_angle)

    def lane_following(self, steer):
        instant_image = self.vision.get_image()
        steering_angle = self.vision.lane_detection_frame(instant_image)
        if steering_angle is None:
            steering_angle = steer
        self.robot.drive(steering_angle, 25)
        return steering_angle


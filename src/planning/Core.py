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

import cv2
from ProcessOnGUI import ProcessOnGUI

class ProcessForExampleImage():
    def __init__(self):
        example_img_name = "./InputImageSet/rust2_large.jpg"
        self.example_img = cv2.imread(example_img_name)
        
        self.process_on_gui = ProcessOnGUI(self.example_img)

    def run_processes(self):
        self.process_on_gui.supervise_GUI_for_example_image()

if __name__ == "__main__":
    process_for_example_image = ProcessForExampleImage()
    process_for_example_image.run_processes()


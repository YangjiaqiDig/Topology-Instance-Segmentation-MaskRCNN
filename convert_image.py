from PIL import Image, ImageDraw
import numpy as np

type_info_dict = {
            0: ("nolabe", (0, 0, 0)),  # no label
            1: ("neopla", (255, 0, 0)),  # neoplastic
            2: ("inflam", (0, 255, 0)),  # inflamm
            3: ("connec", (0, 0, 255)),  # connective
            4: ("necros", (255, 255, 0)),  # dead
            5: ("no-neo", (255, 165, 0)),  # non-neoplastic epithelial
        }
def convert_class_into_color_image_and_save(type_map, save_dir):
        img_array = []
        w, h = type_map.shape
        for x in type_map.reshape(w*h):
            if x == 1.0:
                img_array.append((255,0,0)) # RED
            elif x == 2.0:
                img_array.append((0,255,0)) # GREEN
            elif x == 3.0:
                img_array.append((0,0,255)) # BLUE
            elif x == 4.0:
                img_array.append((255, 255, 0)) # BLACK
            elif x == 5.0:
                img_array.append((255, 165, 0)) # WHITE
            elif x == 0.0:
                img_array.append((0, 0, 0)) # WHITE

        img = Image.new('RGB',(250,250))
        img.putdata(img_array)
        img.save('testImages/{}.png'.format(save_dir))

def draw_box_on_instance_array(img, rec, path):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    [xmin, xmax, ymin, ymax] = rec
    draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=None) 
    convert_class_into_color_image_and_save(np.array(img), path)
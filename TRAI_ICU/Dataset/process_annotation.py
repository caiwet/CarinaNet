import json
import pandas as pd
import os 
def process_annotation(hospital_name):
    root = f"/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/hospital_downsized_new/{hospital_name}/"
    anno_file = root + 'annotations/annotations.json'
    target_file = f"/home/cat302/ETT-Project/CarinaNet/TRAI_ICU/Dataset/{hospital_name}/annotations.json"    
    res = {}
    with open(anno_file, "r") as f:
        data = json.load(f)
    id_to_file = {}
    for image in data['images']:
        id_to_file[image['id']] = image['file_name']
    for anno in data['annotations']:
        file = id_to_file[anno['image_id']]
        image_key = root+'images/'+file
        midpoint_x = anno['bbox'][0] + anno['bbox'][2]/2
        midpoint_y = anno['bbox'][1] + anno['bbox'][3]/2
        if image_key not in res.keys():
            res[image_key] = {
                "CARINA": [None, None],
                "ETT": [None, None],
                "qualite": 2
            }
        if anno['category_id'] == 3046:
            res[image_key]['CARINA'] = [midpoint_x, midpoint_y]
        elif anno['category_id'] == 3047:
            res[image_key]['ETT'] = [midpoint_x, midpoint_y]
        else:
            continue
    with open(target_file, "w") as f:
        json.dump(res, f)

def process_pixel_spacing(hospital_name, default_pixel_spacing=0.125):
    pixel_spacing = pd.read_csv('/home/cat302/ETT-Project/ETT_Evaluation/pixel_spacing_10_hospitals_cleaned.csv')
    new_pixel_spacing = []
    root = f"/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/hospital_downsized_new/{hospital_name}/"
    for image in os.listdir(root+'images'):
        tmp = pixel_spacing[pixel_spacing['image']==image.split('.')[0]]
        if len(tmp) > 0:
            new_pixel_spacing.append([image, tmp['pixel_spacing_x'].values[0]])
        else:
            print("Image has no pixel spacing: ", image)
            new_pixel_spacing.append([image, default_pixel_spacing])
    new_pixel_spacing = pd.DataFrame(new_pixel_spacing)
    new_pixel_spacing.to_csv(f'pixel_spacing_{hospital_name}.csv', header=False, index=False)


if __name__=="__main__":
    hospital_name = 'Cedars-Sinai'
    process_annotation(hospital_name)
    process_pixel_spacing(hospital_name)
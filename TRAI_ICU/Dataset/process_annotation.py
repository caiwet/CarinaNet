import json
import pandas as pd
import os 
def process_annotation(hospital_name):
    root = f"/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/hospital/{hospital_name}/"
    anno_file = root + 'annotations/annotations.json'
    target_folder = f"/n/scratch3/users/c/cat302/ETT-Project/CarinaNet/TRAI_ICU/Dataset/{hospital_name}"
    target_file = target_folder + "/annotations.json"    
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
    ## Delete images without ETT
    remove = []
    for image_key in res.keys():
        if None in res[image_key]['CARINA']:
            print("No carina: ", image_key)
        if None in res[image_key]['ETT']:
            remove.append(image_key)
    for key in remove:
        del res[key] 
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    with open(target_file, "w") as f:
        json.dump(res, f)

def process_pixel_spacing(hospital_name, pixel_spacing_file='/home/cat302/ETT-Project/data_tools/pixel_spacing_17_hospitals.csv', default_pixel_spacing=0.125):
    pixel_spacing = pd.read_csv(pixel_spacing_file)
    new_pixel_spacing = []
    root = f"/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/hospital/{hospital_name}/"
    for image in os.listdir(root+'images'):
        tmp = pixel_spacing[pixel_spacing['image']==image.split('.')[0]]
        if len(tmp) > 0:
            new_pixel_spacing.append([image, tmp['pixel_spacing_x'].values[0]])
        else:
            print("Image has no pixel spacing: ", image)
            new_pixel_spacing.append([image, default_pixel_spacing])
    new_pixel_spacing = pd.DataFrame(new_pixel_spacing)
    target_folder = f"/n/scratch3/users/c/cat302/ETT-Project/CarinaNet/TRAI_ICU/Dataset/{hospital_name}"
    new_pixel_spacing.to_csv(os.path.join(target_folder, f'pixel_spacing_{hospital_name}.csv'), header=False, index=False)


if __name__=="__main__":
    # Original 10
    hospitals = ['Cedars-Sinai','Chiang_Mai_University', 'Morales_Meseguer_Hospital',
                 'Newark_Beth_Israel_Medical_Center', 'NYU_Langone_Health',
                 'Osaka_City_University', 'Technical_University_of_Munich',
                 'Universitätsklinikum_Tübingen', 'University_of_Miami']
    # hospital_name = 'Austral' ## no pixel spacing for now
    # New 7 (no pixel spacing yet)
    # hospitals = ['Ascension-Seton', 'Fundación_Santa_Fe_de_Bogotá', 'Lawson_Health',
    #              'National_University_of_Singapore', 'Rhode_Island_Hospital',
    #              'Sunnybrook_Research_Institute', 'Universitätsklinikum_Essen']
    for hospital_name in hospitals:
        print("Processing: ", hospital_name)
        process_annotation(hospital_name)
        process_pixel_spacing(hospital_name, pixel_spacing_file='/home/cat302/ETT-Project/data_tools/pixel_spacing_17_hospitals.csv',)
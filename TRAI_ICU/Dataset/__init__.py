from Dataset import Dataset

#pth='E:/data/'
pth='/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/hospital_downsized_new/'


# =============================================================================
# dataset = Dataset.Dataset(name= 'MIMIC_CXR_test',
#                           path_to_img = f'{pth}MIMIC-CXR/TEST_SET/mimic_CXR_test/',
#                           annoted = True, #set to True to use the annotations in Dataset/{name}/annotations.json
#                           path_to_pixel_spacing=f'{pth}MIMIC-CXR/TEST_SET/pixel_spacing.csv'
#                           )
# 
# =============================================================================

hospital = 'Cedars-Sinai'
dataset = Dataset.Dataset(name= hospital,
                          path_to_img = f'{pth}{hospital}/images',
                          annoted = True, #set to True to use the annotations in Dataset/{name}/annotations.json
                        #   xls_annot_path = f'{pth}{hospital}/data/TRAI_ICU_v3.xlsx',
                        #   pixel_to_mm=0.2,
                          path_to_pixel_spacing = '/home/cat302/ETT-Project/CarinaNet/TRAI_ICU/Dataset/pixel_spacing_Cedars-Sinai.csv',
                          INFERENCE_MODE=False
                          )


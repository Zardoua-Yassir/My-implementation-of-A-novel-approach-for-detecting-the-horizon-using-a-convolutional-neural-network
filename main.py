from JeongHorizonAlg2 import CnnModelJeong
import os


model_pars_folder_path = r""
model_pars_file = r""

model_pars_file_path = os.path.join(model_pars_folder_path, model_pars_file)  # the CNN's model path
Jeong2_horizon = CnnModelJeong(model_pars_file_path)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['RESULTS_FOLDER'] = "./nnUNet_trained_models/"
import os.path
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
import csv
import sys
import os
from os import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from util.utils import load_itk_image_with_sampling, crop_image_via_box, save_itk_with_backsampling
import SimpleITK as sitk
import numpy as np
import shutil
from natsort import natsorted
from models_nnunet_liver.process import locate_liver_boundingbox, nnUNet_liver_predict
import pandas as pd
from models.liver_staging_model import LiverStagingModel


args_dict = {
'task_name' : '134',
'trainer_class_name' : default_trainer,
'cascade_trainer_class_name' : default_cascade_trainer,
'model' : "3d_fullres",
'plans_identifier' : default_plans_identifier,
'folds' : ['all'],
'save_npz' : False,
'lowres_segmentations' : 'None',
"part_id" : 0,
"num_parts" : 1,
"num_threads_preprocessing" : 6,
"num_threads_nifti_save" : 2,
"disable_tta": False,
"overwrite_existing" : False,
"mode" : "normal",
"all_in_gpu" : "None",
"step_size" : 0.5,
'chk' : 'model_final_checkpoint',
'disable_mixed_precision': False,
}


def load_itk_image_new(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    numpySpacing = list(reversed(itkimage.GetSpacing()))
    numpyDirection = list(reversed(itkimage.GetDirection()))
    return numpyImage, numpyOrigin, numpySpacing, numpyDirection


def save_itk_new(image, filename, origin, spacing, direction):
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    if type(direction) != tuple:
        if type(direction) == list:
            direction = tuple(reversed(direction))
        else:
            direction = tuple(reversed(direction.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, filename, True)


def export_history(header, value, folder, file_name):
    # If folder does not exists make folder
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = folder + file_name
    file_existence = os.path.isfile(file_path)
    # If there is no file make file
    if file_existence == False:
        file = open(file_path, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    # If there is file overwrite
    else:
        file = open(file_path, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(value)
    # Close file when it is done with writing
    file.close()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def csv_to_xlsx_pd(a, b):
    csv = pd.read_csv(a, encoding='utf-8')
    csv.to_excel(b, sheet_name='data')


if __name__ == "__main__":

    print("=" * 30)
    print("Running segmentation:")
    mkdir('./sample_seg_132')

    file_path = r' '
    file_path_list = os.listdir(file_path)
    file_path_list = [item for item in file_path_list if "Vendor" in item]

    for Vendor_rootdir in file_path_list:
        rootdir = path.join(file_path, Vendor_rootdir)
        rootdir_list = os.listdir(rootdir)
        rootdir_list = [item for item in rootdir_list if "_" not in item]
        for f_case in rootdir_list:
            phasedir = path.join(rootdir, f_case)
            phasedir_list = os.listdir(phasedir)
            phasedir_list = [item for item in phasedir_list if "._" not in item]
            phasedir_list = [item for item in phasedir_list if "Store" not in item]
            file_nii = [item for item in phasedir_list if "mask" not in item]

            LiverStaging = LiverStagingModel()
            casename_all_nii_pred = []
            casename_all_nii_pred_S1vsS234 = []
            casename_last = 'first_name'

            for f in file_nii:

                casename = f
                casename_last = casename

                in_path = os.path.join(phasedir, f)

                liver_out_path = os.path.join('./sample_seg_132', f)
                image_sitk, image, origin, spacing, new_spacing, direction, size = \
                    load_itk_image_with_sampling(in_path,spacing=[1.5, 1.5, 1.5])

                nnUNet_lobe_in_dir = './models_nnunet_liver/nnUNet_input'
                nnUNet_lobe_out_dir = './models_nnunet_liver/nnUNet_output'

                if os.path.exists(nnUNet_lobe_in_dir):
                    shutil.rmtree(nnUNet_lobe_in_dir)
                mkdir(nnUNet_lobe_in_dir)

                save_itk_new(image, os.path.join(nnUNet_lobe_in_dir, 'lobe_0000.nii.gz'), origin, tuple(new_spacing),
                             direction)

                nnUNet_liver_predict(nnUNet_lobe_in_dir, nnUNet_lobe_out_dir)

                pred_filelist = natsorted(os.listdir(nnUNet_lobe_out_dir))

                for file in pred_filelist:
                    if file.endswith('nii.gz'):

                        lobe_nnUnet, _, _, _ = load_itk_image_new(os.path.join(nnUNet_lobe_out_dir, file))
                        save_itk_with_backsampling(lobe_nnUnet, liver_out_path, origin, new_spacing, spacing, direction,
                                                   size, islabel=True)

                        if np.sum(lobe_nnUnet) == 0:
                            with open('./task132_seg_nothing_cases_val.txt', 'a') as t:
                                t.write(f)
                                t.write('\r\n')
                        else:
                            lunglobe_boundingbox = locate_liver_boundingbox(lobe_nnUnet)
                            image_crop = crop_image_via_box(image, lunglobe_boundingbox)
                            seg_crop = crop_image_via_box(lobe_nnUnet, lunglobe_boundingbox)

                            new_origin = []
                            i = 0
                            for n in origin:
                                new_origin.append(n + 1.5 * lunglobe_boundingbox[i, 0])
                                i = i + 1

                            img0 = image_crop
                            mean = np.mean(img0)
                            std_dev = np.std(img0)
                            img0 = (img0 - mean) / std_dev
                            img2 = seg_crop
                            img_2ch = np.stack((img0, img2), axis=3)  # check 是不是 3

                            liver_staging = LiverStaging.predict(img_2ch)
                            liver_staging_S123vsS4 = liver_staging[3]
                            liver_staging_S1vsS234 = liver_staging[0]

                            header = ['Case', 'Task1_prob_S4', 'Task2_prob_S1', 'Case_each']
                            values = [f_case + '_' + f, liver_staging_S123vsS4, liver_staging_S1vsS234, f]
                            export_history(header, values, './', "LiFS_pred_all_nii" + ".csv")

                            casename_all_nii_pred.append(liver_staging_S123vsS4)
                            casename_all_nii_pred_S1vsS234.append(liver_staging_S1vsS234)

                if f == file_nii[-1]:
                    header = ['Case', 'Task1_prob_S4', 'Task2_prob_S1']
                    values = [f_case, np.mean(casename_all_nii_pred), np.mean(casename_all_nii_pred_S1vsS234)]
                    export_history(header, values, './', "LiFS_pred" + ".csv")

    csv_to_xlsx_pd('./LiFS_pred.csv', './LiFS_pred_1.xlsx')
    df = pd.read_excel('./LiFS_pred_1.xlsx')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.to_excel('./LiFS_pred.xlsx', index=False)

    print("Done.")




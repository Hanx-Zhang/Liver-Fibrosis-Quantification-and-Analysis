import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['RESULTS_FOLDER'] = "./nnUNet_trained_models/"
import os.path
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from batchgenerators.utilities.file_and_folder_operations import join, isdir
import os
from os import path
from pathlib import Path
import numpy as np
import torch
import os.path
import SimpleITK as sitk
import shutil


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

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def nnUNet_predict(nnUNet_in_dir, nnUNet_out_dir):

    print('==== start nnUNet path ====')
    ## re-define the input folders
    input_folder = nnUNet_in_dir
    if os.path.exists(nnUNet_out_dir):
        shutil.rmtree(nnUNet_out_dir)
    mkdir(nnUNet_out_dir)
    output_folder = nnUNet_out_dir

    part_id = args_dict['part_id']
    num_parts = args_dict['num_parts']
    folds = args_dict['folds']
    save_npz = args_dict['save_npz']
    lowres_segmentations = args_dict['lowres_segmentations']
    num_threads_preprocessing = args_dict['num_threads_preprocessing']
    num_threads_nifti_save = args_dict['num_threads_nifti_save']
    disable_tta = args_dict['disable_tta']
    # print(disable_tta)
    step_size = args_dict['step_size']
    overwrite_existing = args_dict['overwrite_existing']
    mode = args_dict['mode']
    all_in_gpu = args_dict['all_in_gpu']
    model = args_dict['model']
    trainer_class_name = args_dict['trainer_class_name']
    cascade_trainer_class_name = args_dict['cascade_trainer_class_name']
    task_name = args_dict['task_name']

    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)

    assert model in ["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or " \
                                                                             "3d_cascade_fullres"

    if lowres_segmentations == "None":
        lowres_segmentations = None

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    # we need to catch the case where model is 3d cascade fullres and the low resolution folder has not been set.
    # In that case we need to try and predict with 3d low res first
    if model == "3d_cascade_fullres" and lowres_segmentations is None:
        print("lowres_segmentations is None. Attempting to predict 3d_lowres first...")
        assert part_id == 0 and num_parts == 1, "if you don't specify a --lowres_segmentations folder for the " \
                                                "inference of the cascade, custom values for part_id and num_parts " \
                                                "are not supported. If you wish to have multiple parts, please " \
                                                "run the 3d_lowres inference first (separately)"
        model_folder_name = join(network_training_output_dir, "3d_lowres", task_name, trainer_class_name + "__" +
                                 args_dict['plans_identifier'])
        assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name
        lowres_output_folder = join(output_folder, "3d_lowres_predictions")
        predict_from_folder(model_folder_name, input_folder, lowres_output_folder, folds, False,
                            num_threads_preprocessing, num_threads_nifti_save, None, part_id, num_parts,
                            not disable_tta,
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not args_dict['disable_mixed_precision'],
                            step_size=step_size)
        lowres_segmentations = lowres_output_folder
        torch.cuda.empty_cache()
        print("3d_lowres done")

    if model == "3d_cascade_fullres":
        trainer = cascade_trainer_class_name
    else:
        trainer = trainer_class_name

    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" +
                             args_dict['plans_identifier'])
    # print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not args_dict['disable_mixed_precision'],
                        step_size=step_size, checkpoint_name=args_dict['chk'])



def GetConnectedCompont(filename, filename_new):

    sitk_image = sitk.ReadImage(filename)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_mask = cc_filter.Execute(sitk_image)
    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)
    num_connected_label = cc_filter.GetObjectCount()
    area_max_label = 0
    area_max = 0

    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)
        if area > area_max:
            area_max_label = i
            area_max = area

    np_output_mask = sitk.GetArrayFromImage(output_mask)
    res_mask = np.zeros_like(np_output_mask)
    res_mask[np_output_mask == area_max_label] = 1
    res_itk = sitk.GetImageFromArray(res_mask)
    res_itk.SetOrigin(sitk_image.GetOrigin())
    res_itk.SetSpacing(sitk_image.GetSpacing())
    res_itk.SetDirection(sitk_image.GetDirection())
    sitk.WriteImage(res_itk, filename_new, True)


class LiSeg():
    def __init__(self):
        output_path = Path('')
    def predict(self):
        nnUNet_in_dir = './nnUNet_input'
        nnUNet_out_dir = './nnUNet_output'
        nnUNet_predict(nnUNet_in_dir, nnUNet_out_dir)


if __name__ == "__main__":

    print("=" * 30)
    print("Running segmentation:")
    input_dir_path = './nnUNet_input'
    if not os.path.isdir(input_dir_path):
        os.makedirs(input_dir_path)
    out_dir_path = './nnUNet_output'
    if not os.path.isdir(out_dir_path):
        os.makedirs(out_dir_path)
    out_dir_path_final = './nnUNet_output_final'
    if not os.path.isdir(out_dir_path_final):
        os.makedirs(out_dir_path_final)

    file_path = ' '
    file_path_list = os.listdir(file_path)
    file_path_list = [item for item in file_path_list if "Vendor" in item]

    for Vendor_rootdir in file_path_list:
        rootdir = path.join(file_path, Vendor_rootdir)
        rootdir_list = os.listdir(rootdir)
        rootdir_list = [item for item in rootdir_list if "_" not in item]
        for f in rootdir_list:
            phasedir = path.join(rootdir, f)
            phasedir_list = os.listdir(phasedir)
            file = [item for item in phasedir_list if "GED4" in item]
            if len(file) > 0:
                path_old = path.join(phasedir, "GED4.nii.gz")
                path_new = path.join(input_dir_path, 'GED4_0000.nii.gz')
                shutil.copyfile(path_old, path_new)
                print(f"Segmenting image" + path_old)
                LiSeg().predict()
                phasedir = './nnUNet_output/GED4.nii.gz'
                path_new = path.join(out_dir_path_final, f + '_pred_mask_GED4.nii.gz')
                GetConnectedCompont(phasedir, path_new)

    print("Done.")




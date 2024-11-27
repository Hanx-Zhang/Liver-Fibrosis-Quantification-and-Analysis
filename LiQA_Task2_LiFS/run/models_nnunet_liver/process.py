
import os
os.environ['RESULTS_FOLDER'] = "./nnUNet_trained_models/"
import numpy as np
import torch
import os.path
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import shutil


args_dict = {
'task_name' : '132',
'trainer_class_name' : default_trainer,
'cascade_trainer_class_name' : default_cascade_trainer,
'model' : "3d_fullres",
'plans_identifier' : default_plans_identifier,
'folds' : 'None',
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


def locate_liver_boundingbox(image):
    xx, yy, zz = np.where(image)
    airway_boundingbox = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    margin = 2
    airway_boundingbox = np.vstack([np.max([[0, 0, 0], airway_boundingbox[:, 0] - margin], 0),
                                    np.min([np.array(image.shape), airway_boundingbox[:, 1] + margin], axis=0).T]).T
    return airway_boundingbox


def nnUNet_liver_predict(nnUNet_in_dir, nnUNet_out_dir):

    print('==== start nnUNet path ====')
    ## re-define the input folders
    input_folder = nnUNet_in_dir
    if os.path.exists(nnUNet_out_dir):
        shutil.rmtree(nnUNet_out_dir)
    mkdir(nnUNet_out_dir)
    output_folder = nnUNet_out_dir
    # mkdir(output_folder)

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


if __name__ == '__main__':
    print('test')
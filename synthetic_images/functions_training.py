import os
import numpy as np
import ultralytics
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

import functions_annotations
import functions_images

def array_to_string(array):
    return np.array2string(array, separator=',', max_line_width=np.inf)

def string_to_array(string):
    return np.fromstring(string.strip('[]'), sep=',')

def overlay_and_train(input_images_folder, input_overlay_dir, number_of_images_to_overlay,
                      output_overlayed_images_dir, output_overlayed_labels_dir, output_training_dir, nc=1,
                      class_names=None, pre_trained_weights=None, training_name='train', imgsz=800, epochs=30, real_data_yaml_file=None, conf_threshold=.2):

    overlay_result = functions_images.overlay_images(0, input_images_folder, input_overlay_dir,
                                                     output_overlayed_images_dir, output_overlayed_labels_dir,
                                                     max_images=number_of_images_to_overlay,
                                                     number_of_selection_rounds=1, overlay_h_flip_probability=.5,
                                                     overlay_resize_range=(.8, 1.2), overlay_rotation_range=(-180, 180),
                                                     opacity_range=(1, 1),
                                                     brightness_adjustment_factor_range=(0.0, 0.6),
                                                     color_adjustment_factor_range=(0.0, .4))

    functions_annotations.split_data(output_overlayed_images_dir, output_overlayed_labels_dir, output_training_dir)

    if class_names is None:
        class_names = ['ball']

    yaml_data_file = functions_annotations.create_data_yaml(output_training_dir, nc, class_names)

    model = YOLO(pre_trained_weights)

    training_results = model.train(data=yaml_data_file, epochs=epochs, patience=int(epochs / 5) * int(epochs > 20), imgsz=imgsz,
                          verbose=True, name=training_name, seed=42,
                          workers=0)  # workers=0 previene error An attempt has been made to start a new process before the current process has finished its bootstrapping phase.

    if real_data_yaml_file is not None:
        real_data_val_results = model.val(data=real_data_yaml_file, split='val', plots=True, save_json=True, conf=conf_threshold,
                        workers=0)  # workers=0 previene error An attempt has been made to start a new process before the current process has finished its bootstrapping phase.

        real_data_test_results = model.val(data=real_data_yaml_file, split='test', plots=True, save_json=True, conf=conf_threshold,
                        workers=0)  # workers=0 previene error An attempt has been made to start a new process before the current process has finished its bootstrapping phase.

    else:
        real_data_val_results = None
        real_data_test_results = None

    return model, training_results, real_data_val_results, real_data_test_results


def synthetic_trainings(scenarios_to_train, parent_folder="./", real_data_yaml_file=None):
    filest_timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")

    results_by_train = pd.DataFrame(
        columns=['scenario','save_dir', 'train_features_number', 'val_features_number',
                 'start_time', 'end_time', 'train_time', 'precision', 'recall',
                 'map50','map50_95','cm', 'rdv_precision', 'rdv_recall',
                 'rdv_map50','rdv_map50_95','rdv_cm', 'rdt_precision', 'rdt_recall',
                 'rdt_map50','rdt_map50_95','rdt_cm'])

    for m in scenarios_to_train:
        mtr = scenarios_to_train[m]

        pre_trained_weights = '../../Yolo-Weights/yolov8s.pt'
        input_images_folder = mtr["input_images_folder"]
        input_overlay_dir = mtr["input_overlay_dir"]
        number_of_images_to_overlay = mtr['number_of_images_to_overlay']
        epochs = mtr['epochs']
        imgsz = mtr['imgsz']

        output_overlayed_images_dir = os.path.join(parent_folder, m, "overlayed_images")
        output_overlayed_labels_dir = os.path.join(parent_folder, m, "overlayed_labels")
        output_training_dir = os.path.join(parent_folder, m, "training")

        os.makedirs(os.path.join(parent_folder, m), exist_ok=True)
        os.makedirs(output_overlayed_images_dir, exist_ok=True)
        os.makedirs(output_overlayed_labels_dir, exist_ok=True)
        os.makedirs(output_training_dir, exist_ok=True)

        inicio = datetime.now()
        print(inicio, m)
        model, training_results, real_data_val_results, real_data_test_results = overlay_and_train(input_images_folder, input_overlay_dir, number_of_images_to_overlay,
                                    output_overlayed_images_dir, output_overlayed_labels_dir, output_training_dir,
                                    pre_trained_weights=pre_trained_weights, training_name=m, imgsz=imgsz,
                                    epochs=epochs, real_data_yaml_file=real_data_yaml_file)

        train_features_number = len(os.listdir(os.path.join(output_training_dir,'train','images')))
        val_features_number = len(os.listdir(os.path.join(output_training_dir,'val','images')))

        if hasattr(training_results, 'save_dir'):
            save_dir = training_results.save_dir
        else:
            save_dir = ""

        if hasattr(training_results,'results_dict'):
            precision = training_results.results_dict['metrics/precision(B)']
            recall = training_results.results_dict['metrics/recall(B)']
            map50 = training_results.results_dict['metrics/mAP50(B)']
            map50_95 = training_results.results_dict['metrics/mAP50-95(B)']
            cm = array_to_string(training_results.confusion_matrix.matrix.flatten())
        else:
            precision = np.NaN
            recall = np.NaN
            map50 = np.NaN
            map50_95 = np.NaN
            cm = None

        if real_data_val_results is not None and hasattr(real_data_val_results, 'results_dict'):
            rdv_precision = real_data_val_results.results_dict['metrics/precision(B)']
            rdv_recall = real_data_val_results.results_dict['metrics/recall(B)']
            rdv_map50 = real_data_val_results.results_dict['metrics/mAP50(B)']
            rdv_map50_95 = real_data_val_results.results_dict['metrics/mAP50-95(B)']
            rdv_cm = array_to_string(real_data_val_results.confusion_matrix.matrix.flatten())
        else:
            rdv_precision = np.NaN
            rdv_recall = np.NaN
            rdv_map50 = np.NaN
            rdv_map50_95 = np.NaN
            rdv_cm = None

        if real_data_test_results is not None and hasattr(real_data_test_results, 'results_dict'):
            rdt_precision = real_data_test_results.results_dict['metrics/precision(B)']
            rdt_recall = real_data_test_results.results_dict['metrics/recall(B)']
            rdt_map50 = real_data_test_results.results_dict['metrics/mAP50(B)']
            rdt_map50_95 = real_data_test_results.results_dict['metrics/mAP50-95(B)']
            rdt_cm = array_to_string(real_data_test_results.confusion_matrix.matrix.flatten())
        else:
            rdt_precision = np.NaN
            rdt_recall = np.NaN
            rdt_map50 = np.NaN
            rdt_map50_95 = np.NaN
            rdt_cm = None


        conf_threshold = 0.30

        functions_annotations.compare_annotated_images_and_predictions(real_data_yaml_file, 'val', model, os.path.join(parent_folder,m,"error_analysis"),
                                                                       prediction_conf_threshold=conf_threshold)

        fin = datetime.now()
        duracion = fin - inicio

        results_by_train.loc[len(results_by_train.index)] = [m, save_dir, train_features_number, val_features_number,
                   inicio, fin, duracion, precision, recall, map50, map50_95, cm, rdv_precision, rdv_recall, rdv_map50, rdv_map50_95, rdv_cm, rdt_precision, rdt_recall, rdt_map50, rdt_map50_95, rdt_cm]

        results_by_train.to_csv(os.path.join(parent_folder,"results_by_train.csv"))

    return results_by_train


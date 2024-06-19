import numpy as np
import cv2
import os
import re
import random
import yaml
from tqdm import tqdm
from image_video_loader import ImageVideoLoader
import pandas as pd
import matplotlib.pyplot as plt


def rotate_yolo_annotations(annotations, M, img_w, img_h):
    # Annotations are in the format: [class, x_center, y_center, width, height]
    rotated_annotations = []

    for annotation in annotations:
        if len(annotation) == 5:
            cls, x_center, y_center, width, height = annotation

            # Convert normalized coordinates to absolute coordinates
            x_center_abs = x_center * img_w
            y_center_abs = y_center * img_h
            width_abs = width * img_w
            height_abs = height * img_h

            # Get the four corners of the bounding box
            box = np.array([
                [x_center_abs - width_abs / 2, y_center_abs - height_abs / 2],
                [x_center_abs + width_abs / 2, y_center_abs - height_abs / 2],
                [x_center_abs + width_abs / 2, y_center_abs + height_abs / 2],
                [x_center_abs - width_abs / 2, y_center_abs + height_abs / 2]
            ])

            # Rotate the bounding box corners
            ones = np.ones(shape=(len(box), 1))
            points_ones = np.hstack([box, ones])
            rotated_box = M.dot(points_ones.T).T

            # Calculate new bounding box center and size
            x_min, y_min = np.min(rotated_box, axis=0)
            x_max, y_max = np.max(rotated_box, axis=0)

            new_x_center_abs = (x_min + x_max) / 2
            new_y_center_abs = (y_min + y_max) / 2
            new_width_abs = x_max - x_min
            new_height_abs = y_max - y_min

            # Convert back to normalized coordinates
            new_x_center = new_x_center_abs / img_w
            new_y_center = new_y_center_abs / img_h
            new_width = new_width_abs / img_w
            new_height = new_height_abs / img_h

            rotated_annotations.append([cls, new_x_center, new_y_center, new_width, new_height])

    return rotated_annotations


def modify_annotations(annotations, horizontal_resize_factor, vertical_resize_factor=None, h_flipped=False,
                       v_flipped=False):
    modified_annotations = []
    if vertical_resize_factor is None and vertical_resize_factor != 1.0:
        vertical_resize_factor = horizontal_resize_factor

    for annotation in annotations:
        if len(annotation) == 5:
            class_id, x_center, y_center, width, height = annotation

            if vertical_resize_factor is not None:
                x_center *= horizontal_resize_factor
                y_center *= vertical_resize_factor
                width *= horizontal_resize_factor
                height *= vertical_resize_factor

            if h_flipped:
                x_center = .5 + (.5 - x_center)

            if v_flipped:
                y_center = .5 + (.5 - y_center)

            modified_annotations.append([class_id, x_center, y_center, width, height])

    return modified_annotations


def get_annotations_from_file(label_file, height=1, width=1, annotated_classes_to_return=None):
    annotations = []

    with open(label_file, 'r') as file:
        for line in file:
            values = np.array([float(x) for x in line.strip().split()])
            if len(values) == 5:
                class_id, x_center, y_center, box_width, box_height = values
                if annotated_classes_to_return is None or class_id in annotated_classes_to_return:
                    x_center *= width
                    y_center *= height
                    box_width *= width
                    box_height *= height

                    x_min = x_center - box_width / 2
                    y_min = y_center - box_height / 2
                    x_max = x_center + box_width / 2
                    y_max = y_center + box_height / 2

                    annotations.append([int(class_id), x_min, y_min, x_max, y_max])

            elif len(values) > 1:
                class_id = values[0]
                if annotated_classes_to_return is None or class_id in annotated_classes_to_return:
                    points = values[1:].reshape(int(len(values) / 2), 2)

                    x_min = np.min(points, 0)[0] * width
                    y_min = np.min(points, 1)[1] * height
                    x_max = np.max(points, 0)[0] * width
                    y_max = np.max(points, 1)[1] * height

                    annotations.append([int(class_id), x_min, y_min, x_max, y_max])

    return annotations


def get_yolo_annotation_from_xyxy(class_id, xyxy, image_height, image_width):
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w = x2 - x1
    h = y2 - y1
    center_x = x1 + w / 2
    center_y = y1 + h / 2
    center_x /= image_width
    center_y /= image_height
    w /= image_width
    h /= image_height
    return [class_id, center_x, center_y, w, h]


def get_yolo_annotation_from_bbox(class_id, bbox, image_height, image_width):
    x0, y0 = bbox[0]
    x1, y1 = bbox[1]
    x2, y2 = bbox[2]
    x3, y3 = bbox[3]
    x1 = min(x0, x1, x2, x3)
    x2 = max(x0, x1, x2, x3)
    y1 = min(y0, y1, y2, y3)
    y2 = max(y0, y1, y2, y3)
    return get_yolo_annotation_from_xyxy(class_id, (x1, y1, x2, y2), image_height, image_width)


def get_yolo_annotations_from_prediction_results(results, image_height, image_width, classes_to_use=None):
    annotations = []
    for r in results:
        boxes = r.boxes
        for bbox in boxes:
            cls = int(bbox.cls[0])
            if classes_to_use is None or cls in classes_to_use:
                annotations.append([cls, float(bbox.xywh[0][0]), float(bbox.xywh[0][1]), float(bbox.xywh[0][2]),
                                    float(bbox.xywh[0][3]), float(bbox.conf[0])])
    return annotations


def get_xyxy_annotations_from_prediction_results(results, image_height, image_width, classes_to_use=None):
    annotations = []
    for r in results:
        boxes = r.boxes
        for bbox in boxes:
            cls = int(bbox.cls[0])
            if classes_to_use is None or cls in classes_to_use:
                annotations.append([cls, float(bbox.xyxy[0][0]), float(bbox.xyxy[0][1]), float(bbox.xyxy[0][2]),
                                    float(bbox.xyxy[0][3]), float(bbox.conf[0])])
    return annotations


def get_bbox_from_yolo_annotation(values, height=1, width=1):
    if len(values) == 5:
        class_id, x_center, y_center, box_width, box_height = values
        class_id = int(class_id)

        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2
        x_max = x_center + box_width / 2
        y_max = y_center + box_height / 2

    elif len(values) > 1:
        class_id = int(values[0])
        points = values[1:].reshape(int(len(values) / 2), 2)

        x_min = np.min(points, 0)[0] * width
        y_min = np.min(points, 1)[1] * height
        x_max = np.max(points, 0)[0] * width
        y_max = np.max(points, 1)[1] * height
    else:
        class_id = np.NaN
        x_min = np.NaN
        y_min = np.NaN
        x_max = np.NaN
        y_max = np.NaN

    if height > 1 or width > 1:
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    return [class_id, x_min, y_min, x_max, y_max]


def get_annotations_bboxes(annotations, height=1, width=1):
    bboxes = []
    for values in annotations:
        bboxes.append(get_bbox_from_yolo_annotation(values, height, width))
    return bboxes


def get_annotations_bboxes_from_file(label_file, height=1, width=1):
    annotations = get_annotations_from_file(label_file, height, width)
    return annotations


def draw_boxes(image, annotations, color=(0, 255, 0), thickness=2, colors=None, classes=None, label=None):
    for annotation in annotations:
        class_id, x_min, y_min, x_max, y_max = None, None, None, None, None
        if len(annotation) == 5:
            class_id, x_min, y_min, x_max, y_max = annotation
        elif len(annotation) == 6:
            class_id, x_min, y_min, x_max, y_max, conf = annotation
        if x_min is not None:
            if isinstance(colors, dict) and class_id in colors:
                c = colors[class_id]
            else:
                c = color
            if isinstance(classes, dict) and class_id in classes:
                cn = classes[class_id]
            else:
                cn = ""

            if label is None:
                label = cn

            image = cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), c, thickness)
            if label is not None:
                cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def write_annotation_file(annotatios_file, annotations):
    with open(annotatios_file, 'w') as f:
        for line in annotations:
            class_id = line[0]
            x_center = line[1]
            y_center = line[2]
            width_norm = line[3]
            height_norm = line[4]

            f.write(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n")
    f.close()


def show_annotations_in_images(images_dir, labels_dir, authomatic_wait_time=0, color=(0, 255, 0), thickness=2,
                               colors=None, classes=None):
    loader = ImageVideoLoader(images_dir, authomatic_wait_time)
    while len(loader) > 0:
        image, index, image_file = loader.get_element(True)

        label_file = image_file.replace(images_dir, labels_dir).replace(".jpg", ".txt").replace(".jpeg",
                                                                                                ".txt").replace(".png",
                                                                                                                ".txt")

        width, height, _ = image.shape
        bboxes = get_annotations_bboxes_from_file(label_file, width, height)

        annotated_image = draw_boxes(image, bboxes, color, thickness, colors, classes)
        cv2.imshow("Annotated Image", annotated_image)

        if not loader.next_step():
            cv2.destroyAllWindows()
            break


def split_data(input_img_dir, input_label_dir, output_dir, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    input_img_files = os.listdir(input_img_dir)
    random.shuffle(input_img_files)

    num_files = len(input_img_files)
    train_end = int(num_files * train_ratio)
    test_end = train_end + int(num_files * test_ratio)

    train_files = input_img_files[:train_end]
    test_files = input_img_files[train_end:test_end]
    valid_files = input_img_files[test_end:]

    for file_list, folder_name in [(train_files, 'train'), (test_files, 'test'), (valid_files, 'val')]:
        img_output_folder = os.path.join(output_dir, folder_name, 'images')
        label_output_folder = os.path.join(output_dir, folder_name, 'labels')

        os.makedirs(img_output_folder, exist_ok=True)
        os.makedirs(label_output_folder, exist_ok=True)

        for img_file in tqdm(file_list, desc=folder_name):
            img_src = os.path.join(input_img_dir, img_file)
            img_dst = os.path.join(img_output_folder, img_file)

            label_file = img_file.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")
            label_src = os.path.join(input_label_dir, label_file)
            label_dst = os.path.join(label_output_folder, label_file)

            if os.path.exists(img_dst):
                os.remove(img_dst)
            os.rename(img_src, img_dst)

            if os.path.exists(label_src):
                if os.path.exists(label_dst):
                    os.remove(label_dst)
                os.rename(label_src, label_dst)


def get_yaml_data(yaml_file):
    yaml_data = {}
    with open(yaml_file) as stream:
        try:
            yaml_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_data


def get_yaml_value(yaml_data, key):
    value = None
    if key in yaml_data:
        value = yaml_data[key]
    return value


def get_yaml_values_from_file(yaml_file):
    yaml_data = get_yaml_data(yaml_file)
    train = get_yaml_value(yaml_data, "train")
    val = get_yaml_value(yaml_data, "val")
    test = get_yaml_value(yaml_data, "test")
    nc = get_yaml_value(yaml_data, "nc")
    names = get_yaml_value(yaml_data, "names")
    return train, val, test, nc, names


def create_data_yaml(output_dir, nc=1, names=[], file_name='data.yaml'):
    if len(names) == 0:
        names = [x for x in range(0, nc)]
    if len(names) == nc:
        data_yaml = {
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': nc,
            'names': names
        }

        with open(os.path.join(output_dir, file_name), 'w') as yaml_file:
            yaml.dump(data_yaml, yaml_file, default_flow_style=False)

        return os.path.join(output_dir, file_name)
    else:
        return None


# annotations [[cls, x1, y1, x2, y2]...]
# predicted_bboxes [[cls, x1, y1, x2, y2, conf] ... ])
def compare_annotations_predictions(annotations, predictions, iou_threshold_tp=0.5, intersection_min_threshold=0.01):
    prediction_results = []
    annotations_x_predictions = []
    # producto cartesiano de anotaciones y predicciones para calcular iou de cada combinación
    for annotation_number in range(0, len(annotations)):
        acls, ax1, ay1, ax2, ay2 = annotations[annotation_number]
        for pred_number in range(0, len(predictions)):
            if len(predictions[pred_number]) == 6:
                pcls, px1, py1, px2, py2, conf = predictions[pred_number]
            elif len(predictions[pred_number]) == 5:
                pcls, px1, py1, px2, py2 = predictions[pred_number]
                conf = np.NaN
                predictions[pred_number] = predictions[pred_number] + [conf]
            else:
                pcls, px1, py1, px2, py2, conf = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
            if acls == pcls:
                intersection_width = max(0, (px2 - ax1) * (ax2 > px2) * (ax1 > px1))
                intersection_width += max(0, (px2 - px1) * (ax2 > px2) * (ax1 <= px1))
                intersection_width += max(0, (ax2 - px1) * (px2 >= ax2) * (px1 >= ax1))
                intersection_width += max(0, (ax2 - ax1) * (px2 >= ax2) * (px1 < ax1))
                intersection_height = max(0, (py2 - ay1) * (ay2 > py2))
                intersection_height += max(0, (ay2 - py1) * (py2 >= ay2))

                intersection = intersection_width * intersection_height

                union = (ax2 - ax1) * (ay2 - ay1) + (px2 - px1) * (py2 - py1) - intersection

                if intersection >= intersection_min_threshold:
                    annotations_x_predictions.append(
                        [annotation_number, annotations[annotation_number], pred_number, predictions[pred_number],
                         union, intersection, intersection / union])

    # ordena decreciente por iou
    annotations_x_predictions = sorted(annotations_x_predictions, key=lambda x: x[4], reverse=True)

    # busca emparejar anotaciones y predicciones para TP y de las restantes halla FN y FP
    prediction_results = []
    used_annotations = []
    used_predictions = []
    used_results = []
    for a in range(0, len(annotations_x_predictions)):
        axp = annotations_x_predictions[a]
        annotation_number = axp[0]
        pred_number = axp[2]
        iou = axp[6]
        if a not in used_results and annotation_number not in used_annotations and pred_number not in used_predictions:
            if iou > iou_threshold_tp:
                axp.append('TP')
            else:
                axp.append('FN')
            prediction_results.append(axp)
            used_annotations.append(annotation_number)
            used_predictions.append(pred_number)
            used_results.append(a)
    for a in range(0, len(annotations_x_predictions)):
        axp = annotations_x_predictions[a]
        annotation_number = axp[0]
        pred_number = axp[2]
        if a not in used_results and annotation_number not in used_annotations:
            axp[2] = np.NaN
            axp[3] = []
            axp.append('FN')
            prediction_results.append(axp)
            used_annotations.append(annotation_number)
            used_predictions.append(pred_number)
            used_results.append(a)
    for a in range(0, len(annotations_x_predictions)):
        axp = annotations_x_predictions[a]
        pred_number = axp[2]
        if a not in used_results and pred_number not in used_predictions:
            axp[0] = np.NaN
            axp[1] = []
            axp.append('FP')
            prediction_results.append(axp)
            used_predictions.append(pred_number)
            used_results.append(a)
    for annotation_number in range(0, len(annotations)):
        if annotation_number not in used_annotations:
            annotation = annotations[annotation_number]
            axp = [annotation_number, annotation, np.NaN, [],
                   (annotation[3] - annotation[1]) * (annotation[4] - annotation[2]), 0, 0, 'FN']
            prediction_results.append(axp)
            used_annotations.append(annotation_number)
    for pred_number in range(0, len(predictions)):
        if pred_number not in used_predictions:
            prediction = predictions[pred_number]
            axp = [np.NaN, [], pred_number, prediction,
                   (prediction[3] - prediction[1]) * (prediction[4] - prediction[2]),
                   0, 0, 'FP']
            prediction_results.append(axp)
            used_predictions.append(pred_number)
    return prediction_results


def calculate_iou(df):
    df['annotation_x'] = (df['Ax2'] + df['Ax1']) / 2
    df['annotation_y'] = (df['Ay2'] + df['Ay1']) / 2

    df['annotation_width'] = df['Ax2'] - df['Ax1']
    df['annotation_height'] = df['Ay2'] - df['Ay1']

    df['prediction_x'] = (df['Px2'] + df['Px1']) / 2
    df['prediction_y'] = (df['Py2'] + df['Py1']) / 2

    df['prediction_width'] = df['Px2'] - df['Px1']
    df['prediction_height'] = df['Py2'] - df['Py1']

    df['inter_x1'] = df[['Ax1', 'Px1']].max(axis=1)
    df['inter_y1'] = df[['Ay1', 'Py1']].max(axis=1)
    df['inter_x2'] = df[['Ax2', 'Px2']].min(axis=1)
    df['inter_y2'] = df[['Ay2', 'Py2']].min(axis=1)

    df['inter_width'] = (df['inter_x2'] - df['inter_x1']).clip(lower=0)
    df['inter_height'] = (df['inter_y2'] - df['inter_y1']).clip(lower=0)
    df['inter_area'] = df['inter_width'] * df['inter_height']

    df['annotation_area'] = df['annotation_width'] * df['annotation_height']
    df['prediction_area'] = df['prediction_width'] * df['prediction_height']

    df['union_area'] = df['annotation_area'] + df['prediction_area'] - df['inter_area']

    df['iou'] = df['inter_area'] / df['union_area']

    ## calculos normalizados
    df['annotation_x'] = ((df['Ax2'] + df['Ax1']) / 2) / df['Width']
    df['annotation_y'] = ((df['Ay2'] + df['Ay1']) / 2) / df['Height']

    df['annotation_width'] = (df['Ax2'] - df['Ax1']) / df['Width']
    df['annotation_height'] = (df['Ay2'] - df['Ay1']) / df['Height']

    df['prediction_x'] = ((df['Px2'] + df['Px1']) / 2) / df['Width']
    df['prediction_y'] = ((df['Py2'] + df['Py1']) / 2) / df['Height']

    df['prediction_width'] = (df['Px2'] - df['Px1']) / df['Width']
    df['prediction_height'] = (df['Py2'] - df['Py1']) / df['Height']

    return df


def classify_predictions(df, iou_threshold=0.5):
    conditions = [
        (df['iou'] >= iou_threshold),  # True Positive
        (df['iou'] < iou_threshold) & (df['prediction_area'] > 0),  # False Positive
        (df['prediction_area'] == 0)  # False Negative
    ]
    choices = ['TP', 'FP', 'FN']
    df['classification'] = np.select(conditions, choices, default='FN')

    return df


def save_size_heatmap(df_complete, clasification_field='classification', num_steps=10, output_file_name='size_map.png'):
    classifications = df_complete[clasification_field].unique()
    num_classes = len(classifications)

    fig, axes = plt.subplots(1, num_classes, figsize=(12, 4), sharex=True, sharey=True)

    for ax, classification in zip(axes, classifications):
        if classification == 'FP':
            width_field = 'prediction_width'
            height_field = 'prediction_height'
        else:
            width_field = 'annotation_width'
            height_field = 'annotation_height'

        df_filtered = df_complete[df_complete[clasification_field] == classification]
        df_filtered = df_filtered.loc[df_filtered[width_field] > 0]

        df_filtered[width_field] = df_filtered[width_field] / df_filtered['Width']
        df_filtered[height_field] = df_filtered[height_field] / df_filtered['Height']
        if len(df_filtered) > 0:
            min_width = df_filtered[width_field].min()
            max_width = df_filtered[width_field].max()
            min_height = df_filtered[height_field].min()
            max_height = df_filtered[height_field].max()

            annotation_width_bins = np.linspace(min_width, max_width, num_steps + 1)
            annotation_height_bins = np.linspace(min_height, max_height, num_steps + 1)

            counts = np.zeros((num_steps, num_steps))

            for _, row in df_filtered.iterrows():
                annotation_width = row[width_field]
                annotation_height = row[height_field]

                width_bin = np.digitize(annotation_width, annotation_width_bins) - 1
                height_bin = np.digitize(annotation_height, annotation_height_bins) - 1

                if 0 <= height_bin < num_steps and 0 <= width_bin < num_steps:
                    counts[height_bin, width_bin] += 1

            pcm = ax.pcolormesh(counts, cmap='Blues')
            fig.colorbar(pcm, ax=ax)

        ax.set_title(classification)
        if classification == 'FP':
            ax.set_xlabel('Ancho de Predicciones')
            ax.set_ylabel('Alto de Predicciones')
        else:
            ax.set_xlabel('Ancho de Anotaciones')
            ax.set_ylabel('Alto de Anotaciones')

        width_labels = [f'{int(100000 * (width_bins + np.diff(annotation_width_bins)[0])) / 100000}' for width_bins in
                        annotation_width_bins[:-1]]
        height_labels = [f'{int(100000 * (height_bins + np.diff(annotation_height_bins)[0])) / 100000}' for height_bins
                         in
                         annotation_height_bins[:-1]]
        ax.set_xticks(range(num_steps), width_labels, rotation=45)
        ax.set_yticks(range(num_steps), height_labels)

    plt.savefig(output_file_name, bbox_inches='tight')


def save_position_heatmap(df_complete, clasification_field='classification', num_steps=10, output_file_name='position_map.png'):
    classifications = df_complete[clasification_field].unique()

    count_classifications = df_complete[clasification_field].value_counts()

    fig, axes = plt.subplots(1, len(count_classifications), figsize=(12, 4), sharex=True, sharey=True)

    for ax, classification in zip(axes, classifications):
        if classification == 'FP':
            x_field = 'prediction_x'
            y_field = 'prediction_y'
        else:
            x_field = 'annotation_x'
            y_field = 'annotation_y'

        df_filtered = df_complete[df_complete[clasification_field] == classification]
        df_filtered = df_filtered.loc[df_filtered[x_field] > 0]

        if len(df_filtered) > 0:
            min_width = 0
            max_width = 1
            min_height = 0
            max_height = 1

            annotations_x_bins = np.linspace(min_width, max_width, num_steps + 1)
            annotations_y_bins = np.linspace(min_height, max_height, num_steps + 1)

            counts = np.zeros((num_steps, num_steps))

            for _, row in df_filtered.iterrows():
                annotation_x = row[x_field]
                annotation_y = 1 - row[y_field]

                x_bin = np.digitize(annotation_x, annotations_x_bins) - 1
                y_bin = np.digitize(annotation_y, annotations_y_bins) - 1

                if 0 <= y_bin < num_steps and 0 <= x_bin < num_steps:
                    counts[y_bin, x_bin] += 1

            pcm = ax.pcolormesh(counts, cmap='Blues')
            fig.colorbar(pcm, ax=ax)

        ax.set_title(classification)
        if classification == 'FP':
            ax.set_xlabel('X de Predicciones')
            ax.set_ylabel('Y de Predicciones')
        else:
            ax.set_xlabel('X de Anotaciones')
            ax.set_ylabel('Y de Anotaciones')

        width_labels = [f'{int(1000 * (width_bins + np.diff(annotations_x_bins)[0])) / 1000}' for width_bins in
                        annotations_x_bins[:-1]]
        height_labels = [f'{int(1000 * (height_bins + np.diff(annotations_y_bins)[0])) / 1000}' for height_bins in
                         annotations_y_bins[:-1]]
        ax.set_xticks(range(num_steps), width_labels, rotation=45)
        ax.set_yticks(range(num_steps), height_labels)

    plt.savefig(output_file_name, bbox_inches='tight')

def compare_annotated_images_and_predictions(yaml_file, split, model, output_dir, classs_to_use=None,
                                             prediction_conf_threshold=0.30, iou_threshold_tp=0.5,
                                             intersection_min_threshold=0.01, annotation_color=(255, 0, 0),
                                             annotation_thickness=1, prediction_color=(255, 0, 0),
                                             prediction_thickness=1, padding=50):

    #TODO: unificar csv y funciones de clasificacion

    yaml_data = get_yaml_data(yaml_file)

    input_image_dir = os.path.join(os.path.dirname(yaml_file), get_yaml_value(yaml_data, split))

    loader = ImageVideoLoader(input_image_dir, return_element_info=True)

    annotations_and_detections = pd.DataFrame(
        columns=['File1', 'File2', 'Height', 'Width', 'AClass', 'Ax1', 'Ay1', 'Ax2', 'Ay2', 'PClass', 'Px1', 'Py1',
                 'Px2', 'Py2', 'PConf', 'Union', 'Intersection', 'IoU', 'Result'])
    a_a_d_index = 0

    for element in tqdm(loader):
        img, index, image_path = element

        label_path = re.sub('[\\\/]+images[\\\/]+', '/labels/', image_path).replace('.jpg', '.txt')

        image_height, image_width, _ = img.shape

        annotations = get_annotations_from_file(label_path, image_height, image_width)
        results = model.predict(img, conf=prediction_conf_threshold, verbose=False)
        predictions = get_xyxy_annotations_from_prediction_results(results, image_height,
                                                                   image_width,
                                                                   classs_to_use)

        img = draw_boxes(img, annotations, annotation_color, annotation_thickness)
        img = draw_boxes(img, predictions, prediction_color, prediction_thickness)

        comparisons = compare_annotations_predictions(annotations, predictions,
                                                      iou_threshold_tp=iou_threshold_tp,
                                                      intersection_min_threshold=intersection_min_threshold)
        counter = {'TP': 0, 'FP': 0, 'FN': 0}

        for result in counter:
            os.makedirs(os.path.join(output_dir, result), exist_ok=True)
            os.makedirs(os.path.join(output_dir, result), exist_ok=True)
            os.makedirs(os.path.join(output_dir, result), exist_ok=True)
        for comparison in comparisons:
            class_id = comparison[0] or comparison[2]
            annotation = comparison[1]
            prediction = comparison[3]
            union = comparison[4]
            intersection = comparison[5]
            iou = comparison[6]
            result = comparison[7]
            aclass, ax1, ay1, ax2, ay2, pclass, px1, py1, px2, py2, pconf = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN

            if len(annotation) == 5:
                aclass, ax1, ay1, ax2, ay2 = annotation
            if len(prediction) == 6:
                pclass, px1, py1, px2, py2, pconf = prediction

            annotations_and_detections = pd.concat([annotations_and_detections, pd.DataFrame(
                {'File1': image_path, 'File2': label_path, 'Height': image_height, 'Width': image_width,
                 'AClass': comparison[0], 'Ax1': ax1, 'Ay1': ay1, 'Ax2': ax2, 'Ay2': ay2, 'PClass': pclass,
                 'Px1': px1,
                 'Py1': py1, 'Px2': px2, 'Py2': py2,
                 'PConf': pconf, 'Union': union, 'Intersection': intersection, 'IoU': iou, 'Result': result},
                index=[a_a_d_index])])
            a_a_d_index += 1

            y_min, y_max, x_min, x_max = None, None, None, None
            if len(annotation) == 5 and len(prediction) == 6:
                x_min = int(max(0, min(annotation[1], annotation[3], prediction[1], prediction[3]) - padding))
                x_max = int(
                    min(image_width, max(annotation[1], annotation[3], prediction[1], prediction[3]) + padding))
                y_min = int(max(0, min(annotation[2], annotation[4], prediction[2], prediction[4]) - padding))
                y_max = int(
                    min(image_height, max(annotation[2], annotation[4], prediction[2], prediction[4]) + padding))
            elif len(annotation) == 5:
                x_min = int(max(0, min(annotation[1], annotation[3]) - padding))
                x_max = int(min(image_width, max(annotation[1], annotation[3]) + padding))
                y_min = int(max(0, min(annotation[2], annotation[4]) - padding))
                y_max = int(min(image_height, max(annotation[2], annotation[4]) + padding))
            elif len(prediction) == 6:
                x_min = int(max(0, min(prediction[1], prediction[3]) - padding))
                x_max = int(min(image_width, max(prediction[1], prediction[3]) + padding))
                y_min = int(max(0, min(prediction[2], prediction[4]) - padding))
                y_max = int(min(image_height, max(prediction[2], prediction[4]) + padding))

            if x_min is not None:
                crop_img = img[y_min:y_max, x_min:x_max]
                cropped_img_file_name = os.path.join(output_dir, result,
                                                     (os.path.basename(image_path)).replace(".jpg", "_" + str(
                                                         class_id) + '_' + result + '_' + str(
                                                         counter[result]) + ".jpg"))
                cv2.imwrite(cropped_img_file_name, crop_img)
                counter[result] += 1

    annotations_and_detections.to_csv(os.path.join(output_dir, "comparisons.csv"))

    print(annotations_and_detections.Result.value_counts())

    annotations_and_detections = calculate_iou(annotations_and_detections)

    annotations_and_detections = classify_predictions(annotations_and_detections, iou_threshold=iou_threshold_tp)

    annotations_and_detections.to_csv(os.path.join(output_dir, "comparisons_classified.csv"), index=False)

    save_size_heatmap(annotations_and_detections, 'Result', num_steps=10, output_file_name=os.path.join(output_dir, "size_heatmap.png"))

    save_position_heatmap(annotations_and_detections, 'Result', num_steps=10, output_file_name=os.path.join(output_dir, "position_heatmap.png"))
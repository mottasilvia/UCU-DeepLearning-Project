import os
import shutil
import cv2
import random
import numpy as np
from tqdm import tqdm
import functions_annotations
import re
import math


def rotate_image(image, angle):
    # Get image dimensions
    (h, w) = image.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    return rotated_image, M


def get_modify_parameters(rotation_range=(0, 0), h_flip_probability=0, v_flip_probability=0, resize_range=(1.0, 1.0)):
    rotation_angle = random.randint(rotation_range[0], rotation_range[1])

    h_flip = random.random() < h_flip_probability
    v_flip = random.random() < v_flip_probability

    resize_factor = resize_range[0] + random.random() * (resize_range[1] - resize_range[0])

    return rotation_angle, h_flip, v_flip, resize_factor


# las anotaciones en imagenes rotadas quedan más grandes que el objeto anotado
def modify_image(image_to_modify, annotations=None, rotation_angle=1.0, h_flip=False, v_flip=False, resize_factor=1.0, blur_radius=0, brightness_factor=1.0, contrast_factor=1.0, saturation_factor=1.0, gray_threshold=0.0):
    #print({'h_flip': h_flip, 'v_flip': v_flip, 'resize_factor': resize_factor, 'rotation_angle': rotation_angle,
    #       'blur_radius': blur_radius, 'brightness_factor': brightness_factor, 'contrast_factor': contrast_factor,
    #       'saturation_factor': saturation_factor, 'gray_threshold': gray_threshold})

    modified_image = image_to_modify.copy()

    # Separar el canal alfa si existe
    if modified_image.shape[2] == 4:
        bgr_image = modified_image[..., :3]
        alpha_channel = modified_image[..., 3]
    else:
        bgr_image = modified_image
        alpha_channel = None

    if annotations is not None:
        new_annotations = annotations.copy()
    else:
        new_annotations = []

    if h_flip:
        bgr_image = cv2.flip(bgr_image, 1)
    if v_flip:
        bgr_image = cv2.flip(bgr_image, 0)

    if blur_radius > 0:
        ksize = (blur_radius * 2 + 1, blur_radius * 2 + 1)
        bgr_image = cv2.GaussianBlur(bgr_image, ksize, 0)

    if brightness_factor != 1.0 or contrast_factor != 1.0:
        bgr_image = bgr_image.astype(np.float32)
        bgr_image = bgr_image * contrast_factor + brightness_factor * 255 * (1 - contrast_factor)
        bgr_image = np.clip(bgr_image, 0, 255).astype(np.uint8)

    if np.random.rand() < gray_threshold:
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    else:
        if saturation_factor!=0.0:
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_image[..., 1] = hsv_image[..., 1] * saturation_factor
            hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)
            bgr_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    mi_height, mi_width, _ = bgr_image.shape

    if resize_factor != 1.0 and resize_factor != 1:
        mi_height = int(mi_height * resize_factor)
        mi_width = int(mi_width * resize_factor)

        bgr_image = cv2.resize(bgr_image, (mi_width, mi_height))
        if alpha_channel is not None:
            alpha_channel = cv2.resize(alpha_channel, (mi_width, mi_height))

    new_annotations = functions_annotations.modify_annotations(new_annotations, horizontal_resize_factor=resize_factor,
                                                               h_flipped=h_flip, v_flipped=v_flip)

    if rotation_angle != 0 and rotation_angle != 0.0:
        bgr_image, rot_mat = rotate_image(bgr_image, rotation_angle)
        new_annotations = functions_annotations.rotate_yolo_annotations(new_annotations, rot_mat, mi_width, mi_height)
        if alpha_channel is not None:
            alpha_channel, _ = rotate_image(alpha_channel, rotation_angle)

    if alpha_channel is not None:
        bgr_image = cv2.merge((bgr_image, alpha_channel))

    return bgr_image, new_annotations


def get_crops(image, amount=1, annotations=None, min_width=640, max_width=1280, maintain_aspect_ratio=True,
              min_height=640, max_height=1280):
    result = []
    height, width, _ = image.shape
    for i in range(0, amount):
        new_image = image.copy()
        new_width = random.randint(min_width, max_width)
        if maintain_aspect_ratio:
            new_height = int(height * new_width / width)
        else:
            new_height = random.randint(min_height, max_height)
        new_x0 = random.randint(0, width - new_width)
        new_y0 = random.randint(0, height - new_height)

        new_annotations = []
        if annotations is not None:
            for annotation in annotations:
                x = int(annotation[1] * width)
                y = int(annotation[2] * height)
                w = int(annotation[3] * width)
                h = int(annotation[4] * height)
                if new_x0 <= x < new_x0 + new_width and new_y0 <= y < new_y0 + new_height:
                    x1 = max(0, x - int(w / 2) - new_x0)
                    x2 = min(new_width, x + int(w / 2) - new_x0)
                    new_x = int((x1 + x2) / 2)
                    new_w = x2 - x1

                    y1 = max(0, y - int(h / 2) - new_y0)
                    y2 = min(new_height, y + int(h / 2) - new_y0)
                    new_y = int((y1 + y2) / 2)
                    new_h = y2 - y1
                    new_annotation = [annotation[0], new_x / new_width, new_y / new_height, new_w / new_width,
                                      new_h / new_height]
                    new_annotations.append(new_annotation)
        result.append([new_image[new_y0:new_y0 + new_height, new_x0:new_x0 + new_width], new_annotations])
    return result


def augment(origin_images_folder, target_images_folder, copy_original=True, modifications_number=0,
            h_flip_probability=0, v_flip_probability=0, resize=(1, 1), rotation=(0, 0), blur_range=(0,0), brightness_range=(1.0, 1.0), contrast_range=(1.0, 1.0), saturation_range=(1.0, 1.0), gray_threshold_range=(0.0, 0.0), crops_number=0, crops_min_width=640,
            crops_max_width=1280,
            crops_maintain_aspect_ratio=True,
            crops_min_height=640, crops_max_height=1280,
            origin_labels_folder=None,
            target_labels_folder=None):
    images_files = os.listdir(origin_images_folder)

    for img_file_name in tqdm(images_files):
        if os.path.isfile(origin_images_folder + img_file_name):
            label_file_name = img_file_name.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")

            if origin_labels_folder is not None:
                label_file = origin_labels_folder + label_file_name
                annotations = functions_annotations.get_annotations_from_file(label_file)
            else:
                annotations = []

            img_file = origin_images_folder + img_file_name
            image = cv2.imread(img_file)

            if copy_original:
                shutil.copy(origin_images_folder + img_file_name, target_images_folder + img_file_name)
                if origin_labels_folder is not None:
                    shutil.copy(origin_labels_folder + label_file_name, target_labels_folder + label_file_name)

            if crops_number > 0:
                crops = get_crops(image, crops_number, annotations=annotations, min_width=crops_min_width,
                                  max_width=crops_max_width, maintain_aspect_ratio=crops_maintain_aspect_ratio,
                                  min_height=crops_min_height, max_height=crops_max_height)

                for m in range(0, len(crops)):
                    h_flip = random.random() < h_flip_probability
                    v_flip = random.random() < v_flip_probability

                    crop_rotation,crop_h_flip, crop_v_flip, crop_resize = get_modify_parameters(rotation_range=rotation,
                                                                                          h_flip_probability=h_flip,
                                                                                          v_flip_probability=v_flip,
                                                                                          resize_range=resize)

                    blur_radius = int(np.random.uniform(blur_range[0], blur_range[1]))
                    brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
                    contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
                    saturation_factor = np.random.uniform(saturation_range[0], saturation_range[1])
                    gray_threshold = np.random.uniform(gray_threshold_range[0], gray_threshold_range[1])

                    modified_image, new_annotations = modify_image(crops[m][0], crops[m][1], h_flip=crop_h_flip, v_flip=crop_v_flip,
                                                                   resize_factor=crop_resize,
                                                                   rotation_angle=crop_rotation, blur_radius=blur_radius, brightness_factor=brightness_factor, contrast_factor=contrast_factor, saturation_factor=saturation_factor, gray_threshold=gray_threshold)

                    cv2.imwrite(target_images_folder + img_file_name.replace(".jpg", "_c" + str(m) + ".jpg"),
                                modified_image)

                    if origin_labels_folder is not None:
                        functions_annotations.write_annotation_file(
                            target_labels_folder + label_file_name.replace(".txt", "_m" + str(m) + ".txt"), new_annotations)

            for m in range(0, modifications_number):
                h_flip = random.random() < h_flip_probability
                v_flip = random.random() < v_flip_probability
                blur_radius = int(np.random.uniform(blur_range[0], blur_range[1]))
                brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
                contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
                saturation_factor = np.random.uniform(saturation_range[0], saturation_range[1])
                gray_threshold = np.random.uniform(gray_threshold_range[0], gray_threshold_range[1])

                if h_flip != 0 or v_flip != 0 or resize != (1, 1) or rotation != (0, 0) or blur_radius > 0 or brightness_factor!=1.0 or contrast_factor!=1.0 or saturation_factor!=1.0 or gray_threshold>0:
                    rotation_angle, h_flip, v_flip, resize_factor = get_modify_parameters(rotation_range=rotation,
                                                                                          h_flip_probability=h_flip,
                                                                                          v_flip_probability=v_flip,
                                                                                          resize_range=resize)

                    modified_image, new_annotations = modify_image(image, annotations, h_flip=h_flip, v_flip=v_flip,
                                                                   resize_factor=resize_factor,
                                                                   rotation_angle=rotation_angle, blur_radius=blur_radius, brightness_factor=brightness_factor, contrast_factor=contrast_factor, saturation_factor=saturation_factor, gray_threshold=gray_threshold)

                    cv2.imwrite(target_images_folder + img_file_name.replace(".jpg", "_m" + str(m) + ".jpg"), modified_image)
                    if origin_labels_folder is not None:
                        functions_annotations.write_annotation_file(
                            target_labels_folder + label_file_name.replace(".txt", "_m" + str(m) + ".txt"), new_annotations)






def calculate_brightness(image):
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calcular el promedio de brillo
    return np.mean(gray_image)


def adjust_brightness(image, target_brightness, intensity=1.0):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray_image)
    ratio = target_brightness / current_brightness

    # Interpolación lineal para ajustar la intensidad
    adjusted_image = cv2.convertScaleAbs(image, alpha=(1 - intensity) + intensity * ratio, beta=0)
    return adjusted_image


def color_transfer(source, target, intensity=1.0):
    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)

    source_mean = source_mean.flatten()
    source_std = source_std.flatten()
    target_mean = target_mean.flatten()
    target_std = target_std.flatten()

    adjusted_target = (target - target_mean) * (source_std / target_std) + source_mean
    adjusted_target = np.clip(adjusted_target, 0, 255).astype(np.uint8)

    # Interpolación lineal para ajustar la intensidad
    result = cv2.addWeighted(target, 1.0 - intensity, adjusted_target, intensity, 0)
    return result


def get_blend_parameters(opacity_range=(1, 1), brightness_adjustment_factor_range=(0, 0),
                         color_adjustment_factor_range=(0, 0)):
    opacity = random.uniform(opacity_range[0], opacity_range[1])

    brightness_adjustment_factor = random.uniform(brightness_adjustment_factor_range[0],
                                                  brightness_adjustment_factor_range[1])
    color_adjustment_factor = random.uniform(color_adjustment_factor_range[0], color_adjustment_factor_range[1])

    return opacity, brightness_adjustment_factor, color_adjustment_factor

# Function to blend images using alpha blending
def blend_images(background, overlay, position, opacity=1, brightness_adjustment_factor=0, color_adjustment_factor=0):
    x, y = position
    y1, y2 = position[1], position[1] + overlay.shape[0]
    x1, x2 = position[0], position[0] + overlay.shape[1]

    # Extraer los canales de color y el canal alfa de la imagen a superponer
    b, g, r, a = cv2.split(overlay)

    overlay_rgb = cv2.merge((b, g, r))

    # Obtener el ROI (Region of Interest) de la imagen del fondo
    rows, cols, _ = overlay.shape[:3]
    roi = background[y:y + rows, x:x + cols]

    # Calcular el promedio de brillo de la región de interés en la imagen del fondo
    roi_brightness = calculate_brightness(roi)

    # Ajustar el brillo de la imagen a superponer
    adjusted_overlay_rgb = adjust_brightness(overlay_rgb, roi_brightness, brightness_adjustment_factor)

    # Igualar el color de la imagen a superponer al ROI del fondo
    adjusted_overlay_rgb = color_transfer(roi, adjusted_overlay_rgb, color_adjustment_factor)

    # Ajustar la transparencia de la imagen a superponer
    a = (a * opacity).astype(np.uint8)

    # Extraer los canales de color y el canal alfa de la imagen a superponer ajustada
    adjusted_overlay_image = cv2.merge(
        (adjusted_overlay_rgb[:, :, 0], adjusted_overlay_rgb[:, :, 1], adjusted_overlay_rgb[:, :, 2], a))

    # Normalizar el canal alfa para usarlo como máscara
    alpha = a / 255.0
    smooth_alpha = alpha

    # Mezclar la imagen a superponer con la imagen del fondo usando la máscara alfa
    for c in range(0, 3):
        roi[:, :, c] = (alpha * adjusted_overlay_image[:, :, c] + (1 - alpha) * roi[:, :, c])

    # Colocar la imagen combinada en la imagen del fondo
    background[y:y + rows, x:x + cols] = roi

    return background


def overlay_image(class_id, original_img_path, overlay_img_paths, output_img_path, output_annotation_path,
                  probability_per_image,
                  max_images, number_of_selection_rounds=1, overlay_h_flip=0, overlay_v_flip=0,
                  overlay_resize_range=(1, 1), overlay_rotation_range=(0, 0), opacity_range=(1, 1),
                  brightness_adjustment_factor_range=(0, 0), color_adjustment_factor_range=(0, 0)):
    original_img = cv2.imread(original_img_path)

    result = []
    height, width, _ = original_img.shape
    overlay_img = original_img.copy()
    selected_images = []

    for t in range(0, number_of_selection_rounds):
        for i in range(0, len(overlay_img_paths)):
            r = random.uniform(0, 1)
            if r < probability_per_image[i]:
                selected_images.append(overlay_img_paths[i])

    if len(selected_images) > max_images:
        selected_images = random.sample(selected_images, max_images)

    annotation_lines = []
    if len(selected_images)>0:
        for i in range(0, len(selected_images)):
            overlay_path = selected_images[i]
            overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

            rotation_angle, h_flip, v_flip, resize_factor = get_modify_parameters(rotation_range=overlay_rotation_range,
                                                                                  h_flip_probability=overlay_h_flip,
                                                                                  v_flip_probability=overlay_v_flip,
                                                                                  resize_range=overlay_resize_range)

            overlay_img, _ = modify_image(overlay_img, h_flip=h_flip, resize_factor=resize_factor,
                                          rotation_angle=rotation_angle)

            overlay_height, overlay_width, _ = overlay_img.shape

            # Random position to overlay
            x = random.randint(0, original_img.shape[1] - overlay_img.shape[1])
            y = random.randint(0, original_img.shape[0] - overlay_img.shape[0])

            annotation_line = [class_id, (x + overlay_width / 2) / width, (y + overlay_height / 2) / height,
                               overlay_width / width,
                               overlay_height / height]

            annotation_lines.append(annotation_line)

            result.append([re.sub(r'(.*/)([^/]+)', r'\2', overlay_path), annotation_line])

            opacity, brightness_adjustment_factor, color_adjustment_factor = get_blend_parameters(opacity_range=opacity_range, brightness_adjustment_factor_range=brightness_adjustment_factor_range,
                                     color_adjustment_factor_range=color_adjustment_factor_range)

            overlayed_img = blend_images(original_img, overlay_img, (x, y), opacity, brightness_adjustment_factor, color_adjustment_factor)

        cv2.imwrite(output_img_path, overlayed_img)

    functions_annotations.write_annotation_file(output_annotation_path, annotation_lines)

    return result


def overlay_images(class_id, input_original_dir, input_overlay_dir, output_image_dir, output_label_dir, max_images,
                   number_of_selection_rounds=1, overlay_h_flip_probability=0, overlay_v_flip_probability=0,
                   overlay_resize_range=(1, 1), overlay_rotation_range=(0, 0), opacity_range=(1, 1),
                   brightness_adjustment_factor_range=(0, 0), color_adjustment_factor_range=(0, 0)):
    selected_images = []

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Find all overlay images
    overlay_img_paths = [
        os.path.join(input_overlay_dir, overlay_file) for overlay_file in os.listdir(input_overlay_dir)
        if overlay_file.endswith(('.png'))
    ]

    nopi = []

    if overlay_img_paths:
        for oi in overlay_img_paths:
            poi = re.findall(r"_(\d{2})\.png", oi)
            if len(poi) == 0:
                nopi = nopi + [1.0]
            else:
                nopi = nopi + [round(int(poi[0]) / 100, 1)]

        for original_file in tqdm(os.listdir(input_original_dir), desc=input_overlay_dir):
            if original_file.endswith(('.jpg', '.jpeg', '.png')):
                original_path = os.path.join(input_original_dir, original_file)

                output_image_path = os.path.join(output_image_dir, original_file)
                output_label_path = os.path.join(output_label_dir, original_file.replace(".jpg", ".txt"))

                selected_images.append([original_file,
                                        overlay_image(class_id, original_path, overlay_img_paths, output_image_path,
                                                      output_label_path, nopi, max_images,
                                                      number_of_selection_rounds=number_of_selection_rounds,
                                                      overlay_h_flip=overlay_h_flip_probability,
                                                      overlay_v_flip=overlay_v_flip_probability,
                                                      overlay_resize_range=overlay_resize_range,
                                                      overlay_rotation_range=overlay_rotation_range, opacity_range=opacity_range,
                   brightness_adjustment_factor_range=brightness_adjustment_factor_range, color_adjustment_factor_range=color_adjustment_factor_range)])

    return selected_images

import cv2
import os


class ImageVideoLoader:
    def __init__(self, source, wait_time=None, key_mapping=None, from_element=0, to_element=None, step=1, return_element_info = False):
        self.key_mapping = key_mapping or {'previous': [ord('j'), ord('a'), ord('4'), 2424832], 'next': [ord('l'), ord('d'), ord('6'), 2555904],
                                           'quit': [ord('q'), 27], 'pause_resume': [ord(' '), ord('s'), ord('k'), ord('5')],
                                           'faster': [ord('w'), ord('i'), ord('8'), 2490368],
                                           'slower': [ord('x'), ord(','), ord(';'), ord('2'), 2621440]}
        self.original_wait_time = wait_time
        self.wait_time = self.original_wait_time
        self.total_elements = 0
        self.source_name = source
        self.return_element_info = return_element_info
        if os.path.isdir(source):
            self.is_video = False
            image_files = sorted([os.path.join(source, f) for f in os.listdir(source) if
                                  f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
            if not image_files:
                raise ValueError("No images found in the directory")
            self.total_elements = len(image_files)
            self.source = image_files
        elif os.path.isfile(source):
            if source.lower().endswith(('.mp4', '.avi', '.mov')):
                self.is_video = True
                self.cap = cv2.VideoCapture(source)
                if not self.cap.isOpened():
                    raise ValueError("Error opening video file")
                self.total_elements = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.source = self.cap
            elif source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.is_video = False
                self.total_elements = 1
                self.source = source
            else:
                raise ValueError("Unsupported file format")
        else:
            raise ValueError("Source must be a directory, video file, or image file")
        self.from_element = min(max(0, from_element),max(0,self.total_elements-1))
        if to_element is None:
            self.to_element = self.total_elements
        else:
            self.to_element = min(to_element, self.total_elements)
        self.current_element_index = int(self.from_element)
        self.step = step

    def __len__(self):
        return self.total_elements

    def __iter__(self):
        self.current_element_index = self.from_element
        return self

    def __next__(self):
        if self.current_element_index < self.to_element:
            result = self.get_element()
            self.current_element_index += self.step
            return result
        else:
            raise StopIteration


    def set_index(self, new_index):
        if not isinstance(new_index, int) and not isinstance(new_index, float):
            return False
        elif new_index < self.from_element:
            self.current_element_index = int(self.from_element)
        else:
            self.current_element_index = min(self.to_element, int(new_index))
        return True

    def get_element_name(self):
        if self.is_video:
            return self.source_name
        else:
            return self.source[self.current_element_index]

    def get_element(self, include_element_info=None):
        if include_element_info is None:
            include_element_info = self.return_element_info
        result = None
        if self.is_video:
            self.source.set(cv2.CAP_PROP_POS_FRAMES, self.current_element_index)
            ret, frame = self.source.read()
            if not ret:
                raise ValueError("Error reading frame")
            result = frame
        else:
            if self.total_elements == 1:
                img = cv2.imread(self.source)
            else:
                img = cv2.imread(self.source[self.current_element_index])
            if img is None:
                raise ValueError("Error reading image")
            result = img
        if not include_element_info:
            return result
        else:
            return result, self.current_element_index, self.get_element_name()

    def next(self):
        if self.current_element_index + self.step < self.to_element:
            self.current_element_index = self.current_element_index + self.step

            #self.current_element_index = int(
            #    min(self.to_element, (self.current_element_index + self.step) % self.total_elements))

            return True
        else:
            return False

    def previous(self):
        if self.current_element_index > self.from_element:
            self.current_element_index = int(
                max(self.from_element, (self.current_element_index - self.step) % self.total_elements))
            return True
        else:
            return False

    def next_step(self):
        if self.wait_time is None:
            if self.next():
                return True
            else:
                self.release()
                return False
        else:
            key = cv2.waitKeyEx(self.wait_time)  & 0xFF
            if key == 255:  # No se presionó ninguna tecla
                self.next()
            else:
                if key != -1:
                    for action, keys in self.key_mapping.items():
                        if key in keys:
                            if action == 'previous':
                                self.wait_time = 0
                                self.previous()
                            elif action == 'next':
                                self.wait_time = 0
                                self.next()
                            elif action == 'pause_resume':
                                if self.wait_time == 0:
                                    self.wait_time = self.original_wait_time
                                else:
                                    self.wait_time = 0
                            elif action == 'faster':
                                self.step = min(self.step + 1, int((self.to_element - self.from_element) / 2))
                                self.wait_time = self.original_wait_time
                            elif action == 'slower':
                                self.step = max(self.step - 1, -int((self.to_element - self.from_element) / 2))
                                self.wait_time = self.original_wait_time
                            elif action == 'quit':
                                self.release()
                                return False
            return True

    def release(self):
        if self.is_video:
            self.source.release()

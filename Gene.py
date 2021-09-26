import random
import cv2
import numpy as np


class DNA:
    def __init__(
        self, bound, img_gradient, brushstrokes_range, canvas=None, sampling_mask=None
    ):
        self.dna_sequence = []
        self.bound = bound

        # CTRLS
        self.min_brush_size = brushstrokes_range[0]  # 0.1 #0.3
        self.max_brush_size = brushstrokes_range[1]  # 0.3 # 0.7
        self.max_brush_count = 4
        self.brush_img_res = 300  # brush image resolution in pixels
        self.padding = int(self.brush_img_res * self.max_brush_size / 2 + 5)

        self.canvas = canvas

        # IMG GRADIENT
        self.img_mag = img_gradient[0]
        self.img_angles = img_gradient[1]

        # OTHER
        self.brushes = [
            cv2.imread("brushes/watercolor/" + str(i) + ".jpg")
            for i in range(self.max_brush_count)
        ]
        self.sampling_mask = sampling_mask

        # CACHE
        self.cached_image = None
        self.cached_error = None

    def randomize_dna(self, target_image, count, seed):
        # initialize random DNA sequence
        for i in range(count):
            # random color
            color = random.randrange(0, 255)
            # random size
            random.seed(seed - i + 4)
            size = (
                random.random() * (self.max_brush_size - self.min_brush_size)
                + self.min_brush_size
            )
            # random pos
            y_pos, x_pos = self.__gen_new_positions()
            # random rotation

            # start with the angle from image gradient
            # based on magnitude of that angle direction, adjust the random angle offset.
            # So in places of high magnitude, we are more likely to follow the angle with our brushstroke.
            # In places of low magnitude, we can have a more random brushstroke direction.

            random.seed(seed * i / 4.0 - 5)
            local_magnitude = self.img_mag[y_pos][x_pos]
            local_angle = self.img_angles[y_pos][x_pos] + 90  # perpendicular to the dir
            rotation = random.randrange(-180, 180) * (1 - local_magnitude) + local_angle
            # random brush number
            brush_number = random.randrange(1, self.max_brush_count)
            # append data
            self.dna_sequence.append(
                [color, y_pos, x_pos, size, rotation, brush_number]
            )
        # calculate cache error and image
        self.cached_error, self.cached_image = self.__calc_wrong_strokes(
            self.dna_sequence, target_image
        )

    def evolve_dna(self, work_image, seed):
        for index, _ in enumerate(self.dna_sequence):
            # create a copy of the list and get its child
            dna_sequence_copy = np.copy(self.dna_sequence)
            child = dna_sequence_copy[index]

            # select which items to mutate and mutate the child
            random.seed(seed + index)
            index_options = [0, 1, 2, 3, 4, 5]
            change_count = random.randrange(1, 7)  # index_options count + 1
            change_indices = [
                index_options.pop(random.randrange(0, len(index_options)))
                for _ in range(change_count)
            ]
            # mutate selected items
            np.sort(change_indices)
            change_indices[:] = change_indices[::-1]
            for change_index in change_indices:
                if change_index == 0:  # if color
                    child[0] = int(random.randrange(0, 255))
                elif change_index in (1, 2):  # if pos Y or X
                    child[1], child[2] = self.__gen_new_positions()
                elif change_index == 3:  # if size
                    child[3] = (
                        random.random() * (self.max_brush_size - self.min_brush_size)
                        + self.min_brush_size
                    )
                elif change_index == 4:  # if rotation
                    local_magnitude = self.img_mag[int(child[1])][int(child[2])]
                    local_angle = (
                        self.img_angles[int(child[1])][int(child[2])] + 90
                    )  # perpendicular
                    child[4] = (
                        random.randrange(-180, 180) * (1 - local_magnitude)
                        + local_angle
                    )
                elif change_index == 5:  # if  brush number
                    child[5] = random.randrange(1, self.max_brush_count)
            child_error, child_img = self.__calc_wrong_strokes(
                dna_sequence_copy, work_image
            )
            if child_error < self.cached_error:
                self.dna_sequence[index] = child[:]
                self.cached_image = child_img
                self.cached_error = child_error

    def __gen_new_positions(self):
        if self.sampling_mask is not None:
            pos = self.__get_sample_from_img(self.sampling_mask)
            y_pos = pos[0][0]
            x_pos = pos[1][0]
        else:
            y_pos = int(random.randrange(0, self.bound[0]))
            x_pos = int(random.randrange(0, self.bound[1]))
        return [y_pos, x_pos]

    def __calc_wrong_strokes(self, dna_sequence, work_image):
        # draw the DNA
        cur_image = self.__draw_image(dna_sequence)
        return (np.sum(cv2.absdiff(work_image, cur_image)), cur_image)

    def __draw_image(self, dna_sequence):
        # set image to pre generated
        if self.canvas is None:  # if we do not have an image specified
            work_image = np.zeros((self.bound[0], self.bound[1]), np.uint8)
        else:
            work_image = np.copy(self.canvas)
        # apply padding
        padding = self.padding
        work_image = cv2.copyMakeBorder(
            work_image,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        # draw every DNA
        for _, cur_dna in enumerate(dna_sequence):
            work_image = self.__draw_dna(cur_dna, work_image)
        # remove padding
        return work_image[
            padding : (work_image.shape[0] - padding),
            padding : (work_image.shape[1] - padding),
        ]

    def __draw_dna(self, worked_dna, worked_image):
        # get worked_dna data
        color = worked_dna[0]
        x_pos = (
            int(worked_dna[2]) + self.padding
        )  # add padding since indices have shifted
        y_pos = int(worked_dna[1]) + self.padding
        size = worked_dna[3]
        rotation = worked_dna[4]
        brush_number = int(worked_dna[5])

        # load brush alpha
        brush_img = self.brushes[brush_number]
        # resize the brush
        brush_img = cv2.resize(
            brush_img, None, fx=size, fy=size, interpolation=cv2.INTER_CUBIC
        )
        # rotate
        # brush img data
        brush_img = cv2.cvtColor(
            self.__rotate_img(brush_img, rotation), cv2.COLOR_BGR2GRAY
        )
        rows, cols = brush_img.shape

        # create a colored canvas
        colored_canvas = np.copy(brush_img)
        colored_canvas[:, :] = color

        # find ROI
        y_min = int(y_pos - rows / 2)
        y_max = int(y_pos + (rows - rows / 2))
        x_min = int(x_pos - cols / 2)
        x_max = int(x_pos + (cols - cols / 2))

        # Convert uint8 to float
        foreground = colored_canvas[0:rows, 0:cols].astype(float)
        background = worked_image[y_min:y_max, x_min:x_max].astype(float)  # get ROI
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = brush_img.astype(float) / 255.0

        try:
            # Multiply the foreground with the alpha matte
            foreground = cv2.multiply(alpha, foreground)
            # Multiply the background with ( 1 - alpha )
            background = cv2.multiply(
                np.core.umath.clip((1.0 - alpha), 0.0, 1.0), background
            )
            # Add the masked foreground and background.
            worked_image[y_min:y_max, x_min:x_max] = (
                np.core.umath.clip(cv2.add(foreground, background), 0.0, 255.0)
            ).astype(np.uint8)
        except:
            print("Error Drawing")
        return worked_image

    def __rotate_img(self, img, angle):
        rows, cols, _ = img.shape
        rotated_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(img, rotated_matrix, (cols, rows))

    def __get_sample_from_img(self, img):
        # possible positions to sample
        pos = np.indices(dimensions=img.shape)
        pos = pos.reshape(2, pos.shape[1] * pos.shape[2])
        return pos[
            :,
            np.random.choice(
                pos.shape[1],
                1,
                p=np.core.umath.clip(img.flatten() / img.flatten().sum(), 0.0, 1.0),
            ),
        ]

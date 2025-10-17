#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 AG-RAIMIA-Müller, University of Augsburg,                #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
from PIL import Image
import cv2
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#       Subfunction class: Stain Normalization        #
#-----------------------------------------------------#
class StainNormalization(Subfunction_Base):
    """ A Subfunction class which which can be used for stain normalization
        of histopathology scans.

    Digital pathology images can present strong color differences due to diverse
    acquisition techniques (e.g., scanners, laboratory equipment and procedures).

    Therefore, this Subfunction class applies Reinhard stain normalization based
    on a provided source image.

    The provided source image should be the same for reproducible results.

    Reference:
        Reinhard, Erik, et al. "Color transfer between images." IEEE Computer graphics and applications 21.5 (2001)

    Implementation:
        Direct implementation of Reinhard color normalization in LAB color space.
        Matches mean and standard deviation of LAB channels between source and target images.
    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, source_image):
        """ Initialization function for creating a StainNormalization Subfunction which can be passed to a
            [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

        Args:
            source_image (Pillow Image):    Pillow image which is used as source for the stain normalization.
        """
        # Convert source image to numpy array
        target_rgb = np.asarray(source_image.convert("RGB")).astype(np.uint8)

        # Convert target image to LAB color space
        target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)

        # Calculate mean and std for each channel
        self.target_mean = np.mean(target_lab, axis=(0, 1))
        self.target_std = np.std(target_lab, axis=(0, 1))

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        """ Apply Reinhard stain normalization to the input image.

        Args:
            image (numpy array): Input RGB image as numpy array

        Returns:
            numpy array: Normalized RGB image
        """
        # Clip and convert image to uint8
        image_uint8 = np.clip(image, a_min=0, a_max=255).astype(np.uint8)

        # Convert to LAB color space
        image_lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Calculate mean and std for source image
        source_mean = np.mean(image_lab, axis=(0, 1))
        source_std = np.std(image_lab, axis=(0, 1))

        # Normalize each channel
        normalized_lab = np.zeros_like(image_lab)
        for i in range(3):
            # Avoid division by zero
            if source_std[i] > 0:
                normalized_lab[:, :, i] = ((image_lab[:, :, i] - source_mean[i]) *
                                          (self.target_std[i] / source_std[i]) +
                                          self.target_mean[i])
            else:
                normalized_lab[:, :, i] = image_lab[:, :, i]

        # Clip LAB values to valid range
        normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)

        # Convert back to RGB
        normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)

        return normalized_rgb

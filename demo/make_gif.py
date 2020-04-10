import glob
import os

import logging
import imageio

def generate_gif(filenames, gif_filename):
        '''create an animated GIF given a list of images

        Args:
            filenames (list): list of image filenames, ordered in required sequence
            gif_filename (str): filepath of final GIF file

        Returns:
            nothing. Side effect is to save a GIF file at gif_filename

        '''
        images = []
        for filename in filenames:
            if os.path.exists(filename):
                logging.info("Adding to gif: image " + filename)
                images.append(imageio.imread(filename))
                # josie.debug
                # print("{} exists.".format(filename))

        logging.info("Creating GIF. This can take some time...")

        imageio.mimsave(gif_filename, images)

        logging.info("Gif generated at " + gif_filename)
		
def main():
	gif_filename = "../img/demo.gif"
	image_names = sorted(glob.glob(os.path.join('./', '*.png')))
	# josie.debug
	# print("get images: \n{}".format(image_names))
	generate_gif(image_names, gif_filename)
	
if __name__ == '__main__':
    main()

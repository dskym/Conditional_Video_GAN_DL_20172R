import moviepy.editor as mpy
import glob
import re

def make_gif(root, output, fps) :
    file_list = glob.glob(root + '/*.png')
    
    file_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    clip = mpy.ImageSequenceClip(file_list, fps=fps)
    clip.write_gif(output, fps=fps)
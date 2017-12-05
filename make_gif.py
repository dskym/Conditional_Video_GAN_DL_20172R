import moviepy.editor as mpy
import glob

def make_gif(root, output, fps) :
    file_list = glob.glob(root + '/*.png')

    clip = mpy.ImageSequenceClip(file_list, fps=fps)
    clip.write_gif(output, fps=fps)

if __name__ == '__main__' :
    make_gif('./test', 'output.gif', 16)
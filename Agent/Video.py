from tqdm import tqdm
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def make_video(vid, filenname='/tmp/video'):
    fig = plt.figure()
    ims = []
    for frame in tqdm(vid):
        im = plt.imshow(frame, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True,
                                    repeat_delay=1000)
    ani.save(name+'.mp4')

if __name__ == '__main__':
    import sys
    vid = torch.load(sys.argv[1])
    make_video(vid, sys.argv[2])

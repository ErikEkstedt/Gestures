from tqdm import tqdm

def make_video(vid, filename='/tmp/video'):

    import matplotlib
    # matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    ims = []
    for frame in tqdm(vid):
        im = plt.imshow(frame, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True,
                                    repeat_delay=1000)
    name = filename+'.mp4'
    ani.save(name)
    print('Saved video to:', name)

def watch_result(vid, args=None):
    import cv2
    name = args.load_file.split('/')[-1]
    print(name)
    while True:
        for v in vid:
            cv2.imshow(name, cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

def watch_result(vid):
    import cv2
    while True:
        for v in vid:
            cv2.imshow('frame', cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(20) & 0xFF == ord('q'):
                return

def watch_folder(result_dir):
    import os
    for d, r, files in os.walk(result_dir):
        print('directory:',d)
        for f in files:
            filepath = os.path.join(d, f)
            print(f)
            vid = torch.load(filepath)
            watch_result(vid)


if __name__ == '__main__':
    # from arguments import get_args
    import torch
    # args = get_args()
    # vid = torch.load(args.load_file)
    # make_video(vid)
    # watch_result(vid, args)

    import sys
    watch_folder(sys.argv[1])

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

def watch_result(args, vid):
    import cv2
    name = args.load_file.split('/')[-1]
    print(name)
    while True:
        for v in vid:
            cv2.imshow(name, cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    from arguments import get_args
    import torch
    args = get_args()
    vid = torch.load(args.load_file)
    # make_video(vid)
    watch_result(args, vid)

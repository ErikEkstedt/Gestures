#####################################################################################################
# From https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
#####################################################################################################
import tensorflow as tf
import numpy as np
import scipy.misc
import torch
import os
import shutil
import datetime
from torch.autograd import Variable
# import torchvision as tv

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


# ---------------------------------
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def transform(imgs):
    return imgs


def get_today():
    t = datetime.date.today().ctime().split()[1:3]
    s = "".join(t)
    return s


class Logger(object):
    def __init__(self, dir, name=None):
        """Create a summary writer logging to log_dir."""
        self.day = get_today()
        if name is not None:
            self.root_dir = dir + '/' + name + '_' + self.day
        else:
            self.root_dir = dir + '/' + self.day
        self.run = 0
        self._mkdirs()

        self.writer = tf.summary.FileWriter(self.log_dir)
        self.best_score = None

    def _mkdirs(self):
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)

        while os.path.exists("%s/run-%d" % (self.root_dir, self.run)):
            self.run += 1

        self.run_dir = "%s/run-%d" % (self.root_dir, self.run)
        self.log_dir = "%s/run-%d/logs" % (self.root_dir, self.run)
        self.checkpoint_dir = "%s/run-%d/checkpoints" % (self.root_dir, self.run)
        os.mkdir(self.run_dir)
        os.mkdir(self.log_dir)
        os.mkdir(self.checkpoint_dir)

    def save_checkpoint(self, model, optimizer, epoch, score, fm='/checkpoint.pth.tar'):
        filename = self.checkpoint_dir + fm
        if self.best_score is None:
            self.best_score = score

        if model.cuda:
            model = model.cpu()

        is_best = score >= self.best_score
        best = max(score, self.best_score)
        state = {'epoch': epoch,
                 'best_prec1': best,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, self.checkpoint_dir + '/model_best.pth.tar')

        if model.cuda:
            model = model.cuda()

    def add_loss(self, loss, step, name='loss'):
        try:
            info = {name: loss.cpu().data[0]}
        except:
            info = {name: loss}

        for tag, value in info.items():
            self.scalar_summary(tag, value, step+1)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def add_images(self, images, outputs, step):
        info = {'inputs': to_np(images),
                'outputs': to_np(outputs)}
        for tag, images in info.items():
            self.image_summary(tag, images, step+1)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")
            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
            height=img.shape[0],
            width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def add_parameter_data(self, net, step):
        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            self.histo_summary(tag, to_np(value), step+1)
            self.histo_summary(tag+'/grad', to_np(value.grad), step+1)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)

    def _flush(self):
        self.writer.flush()


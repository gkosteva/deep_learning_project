import os
import tempfile
import unittest

from src.fake_news.reporting.plots import (plot_class_distribution, plot_confusion_matrix,
                                           plot_text_length_histogram, plot_training_curves)
from src.fake_news.training.trainer import TrainingHistory


class TestPlotClassDistribution(unittest.TestCase):

    def test_when_called_then_image_file_is_created(self):
        path = os.path.join(tempfile.mkdtemp(), 'dist.png')
        plot_class_distribution([0, 0, 1], path)
        self.assertTrue(os.path.exists(path))


class TestPlotTextLengthHistogram(unittest.TestCase):

    def test_when_called_then_image_file_is_created(self):
        path = os.path.join(tempfile.mkdtemp(), 'len.png')
        plot_text_length_histogram(['a b c', 'd e'], path)
        self.assertTrue(os.path.exists(path))


class TestPlotTrainingCurves(unittest.TestCase):

    def test_when_called_then_both_image_files_are_created(self):
        directory = tempfile.mkdtemp()
        history = TrainingHistory(train_loss=[1.0, 0.5],
                                  val_loss=[1.1, 0.6],
                                  train_f1=[0.4, 0.7],
                                  val_f1=[0.3, 0.6])
        metric_path = os.path.join(directory, 'f1.png')
        loss_path = os.path.join(directory, 'loss.png')
        paths = plot_training_curves(history, metric_path, loss_path)
        self.assertTrue(all(os.path.exists(path) for path in paths))


class TestPlotConfusionMatrix(unittest.TestCase):

    def test_when_called_then_image_file_is_created(self):
        path = os.path.join(tempfile.mkdtemp(), 'cm.png')
        plot_confusion_matrix([[5, 1], [2, 4]], path)
        self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()

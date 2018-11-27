"""Extract features based on shapelets
"""

# import click
# import numpy as np
# import shapelets_lts.classification as sl
# from sklearn import metrics

# from ..utils import io, cli


# @click.command("test-shapelet")
# @cli.option_input_file()
# @click.option("--room-count", default=50)
# @click.option("--sensor-count", default=4)
# def test_shapelet(input_file: str, room_count, sensor_count):
#     """Run minimal test on shapelet training with KETI one
#     """
#     data = io.read_file(input_file)
#     labels = np.zeros(data.shape[0])
#     labels[:4] = 1

#     sample_count = data.shape[0]
#     l_min = int(0.2 * sample_count)
#     data = data[:, : l_min * (data.shape[1] // l_min)]

#     classifier = sl.LtsShapeletClassifier(
#         K=int(0.15 * sample_count),
#         R=3,
#         L_min=l_min,
#         epocs=200,
#         lamda=0.01,
#         eta=0.01,
#         shapelet_initialization="segments_centroids",
#         plot_loss=True,
#     )

#     classifier.fit(data, labels)

#     prediction = classifier.predict(data)
#     print(metrics.classification_report(labels, prediction))

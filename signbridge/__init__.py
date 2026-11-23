"""SignBridge research core: dataset-independent data pipeline and model interfaces
for Continuous Sign Language Translation (sign video -> text -> speech).

Layout
------
signbridge.data      dataset adapters (How2Sign, VSL), common manifest schema, batching
signbridge.video     video I/O, frame sampling, transforms, temporal windows
signbridge.features  optional pose (hand keypoint) extraction
signbridge.text      gloss / text vocabularies
signbridge.models    model interfaces + untrained reference modules (shape contracts only)

Dataset-specific logic lives only in ``signbridge.data.<dataset>`` modules; nothing in
``signbridge.video`` or ``signbridge.models`` knows which dataset produced a sample.
"""

__version__ = "0.1.0"

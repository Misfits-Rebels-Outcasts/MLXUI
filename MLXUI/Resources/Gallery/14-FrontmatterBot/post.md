# Why we moved inference on-device

For a year we ran every model call through a hosted API. It was
easy to start with, but the latency budget kept shrinking as the
pipeline grew, and every added step meant another round trip. We
switched to running quantized models locally and cut end-to-end
latency by 70%, at the cost of managing our own model updates.
This post walks through what changed, what broke, and what we'd
do differently.
